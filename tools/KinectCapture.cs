// Kinect v2 SDK bridge. Binary stdout only; diagnostics go to stderr.
using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Kinect;

class KinectCapture {
    static volatile bool stop;
    static int Main() {
        KinectSensor sensor = null;
        try {
            sensor = KinectSensor.GetDefault();
            if (sensor == null) throw new Exception("Kinect v2 was not found. Check power and USB 3.0.");
            sensor.Open();
            new Thread(delegate() { Console.ReadLine(); stop = true; }) { IsBackground = true }.Start();
            using (var reader = sensor.OpenMultiSourceFrameReader(FrameSourceTypes.Color | FrameSourceTypes.Depth | FrameSourceTypes.Body))
            using (var output = new BinaryWriter(Console.OpenStandardOutput())) {
                var bodies = new Body[sensor.BodyFrameSource.BodyCount];
                byte[] rgba = new byte[1920*1080*4];
                ushort[] depth = new ushort[512*424];
                byte[] depthBytes = new byte[512*424*2];
                var xyz = new CameraSpacePoint[depth.Length];
                var uv = new ColorSpacePoint[depth.Length];
                DateTime last = DateTime.UtcNow;
                while (!stop) {
                    if ((DateTime.UtcNow-last).TotalSeconds > 15) throw new Exception("No synchronized Kinect frames for 15 seconds. Check power and USB 3.0.");
                    var multi = reader.AcquireLatestFrame(); {
                        if (multi == null) {
                            if ((DateTime.UtcNow-last).TotalSeconds > 15) throw new Exception("No synchronized Kinect frames for 15 seconds. Check USB 3.0, power and other Kinect applications.");
                            Thread.Sleep(3); continue;
                        }
                        using (var color = multi.ColorFrameReference.AcquireFrame())
                        using (var dep = multi.DepthFrameReference.AcquireFrame())
                        using (var body = multi.BodyFrameReference.AcquireFrame()) {
                            if (color == null || dep == null || body == null) continue;
                            last = DateTime.UtcNow;
                            color.CopyConvertedFrameDataToArray(rgba, ColorImageFormat.Bgra);
                            dep.CopyFrameDataToArray(depth); body.GetAndRefreshBodyData(bodies);
                            sensor.CoordinateMapper.MapDepthFrameToCameraSpace(depth, xyz);
                            sensor.CoordinateMapper.MapDepthFrameToColorSpace(depth, uv);
                            using (var memory = new MemoryStream()) using (var w = new BinaryWriter(memory)) {
                                w.Write(1); w.Write(dep.RelativeTime.TotalSeconds);
                                w.Write(color.RelativeTime.TotalSeconds); w.Write(body.RelativeTime.TotalSeconds);
                                int count = 0; foreach (var b in bodies) if (b != null && b.IsTracked) count++;
                                w.Write(count);
                                foreach (var b in bodies) if (b != null && b.IsTracked) {
                                    w.Write(b.TrackingId);
                                    for (int j=0;j<25;j++) {
                                        var joint=b.Joints[(JointType)j]; var p=joint.Position;
                                        w.Write(p.X);w.Write(p.Y);w.Write(p.Z);w.Write((int)joint.TrackingState);
                                        var q=b.JointOrientations[(JointType)j].Orientation;
                                        w.Write(q.X);w.Write(q.Y);w.Write(q.Z);w.Write(q.W);
                                    }
                                }
                                using (var bitmap = new Bitmap(1920,1080,PixelFormat.Format32bppArgb)) {
                                    var data=bitmap.LockBits(new Rectangle(0,0,1920,1080),ImageLockMode.WriteOnly,PixelFormat.Format32bppArgb);
                                    try { Marshal.Copy(rgba,0,data.Scan0,rgba.Length); } finally { bitmap.UnlockBits(data); }
                                    using (var jpeg=new MemoryStream()) {
                                        bitmap.Save(jpeg,ImageFormat.Jpeg);byte[] bytes=jpeg.ToArray();w.Write(bytes.Length);w.Write(bytes);
                                    }
                                }
                                w.Write(depth.Length);Buffer.BlockCopy(depth,0,depthBytes,0,depthBytes.Length);w.Write(depthBytes);
                                // Registered RGB point cloud in camera metres. Keep invalid pixels out.
                                using(var points=new MemoryStream()) using(var pw=new BinaryWriter(points)) {
                                    int n=0;
                                    for(int y=0;y<424;y+=4) for(int x=0;x<512;x+=4) {
                                        int i=y*512+x;var p=xyz[i];var c=uv[i];
                                        if(depth[i]<dep.DepthMinReliableDistance || depth[i]>dep.DepthMaxReliableDistance || float.IsInfinity(p.Z) || float.IsNaN(p.Z) || float.IsInfinity(c.X) || float.IsNaN(c.X) || float.IsInfinity(c.Y) || float.IsNaN(c.Y)) continue;
                                        int cx=(int)Math.Round(c.X),cy=(int)Math.Round(c.Y);
                                        if(cx<0 || cx>=1920 || cy<0 || cy>=1080) continue;
                                        int k=4*(cy*1920+cx);pw.Write(p.X);pw.Write(p.Y);pw.Write(p.Z);
                                        pw.Write(rgba[k+2]);pw.Write(rgba[k+1]);pw.Write(rgba[k]);pw.Write((byte)255);n++;
                                    }
                                    w.Write(n);w.Write(points.ToArray());
                                }
                                byte[] packet=memory.ToArray(); output.Write(packet.Length);output.Write(packet);output.Flush();
                            }
                        }
                    }
                }
            }
            return 0;
        } catch(Exception e) { Console.Error.WriteLine(e.Message);return 1; }
        finally { if(sensor!=null) sensor.Close(); }
    }
}
