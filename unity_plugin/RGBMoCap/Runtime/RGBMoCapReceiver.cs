using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace RGBMoCap
{
    [Serializable] public class ActorFrame
    {
        public string id, name, rig;
        public int frame;
        public string[] names;
        public int[] parents;
        public float[] rest, pose;
        public Vector3 Position(float[] data, int bone)
        { int i = bone * 12; return new Vector3(data[i + 3], data[i + 7], data[i + 11]); }
        public Quaternion Rotation(float[] data, int bone)
        {
            int i = bone * 12;
            return Quaternion.LookRotation(new Vector3(data[i+2], data[i+6], data[i+10]),
                                           new Vector3(data[i+1], data[i+5], data[i+9]));
        }
    }
    [Serializable] public class StreamFrame { public int version; public ActorFrame[] actors; }

    [DefaultExecutionOrder(10000)]
    [AddComponentMenu("RGB MoCap/Stream Receiver")]
    public sealed class RGBMoCapReceiver : MonoBehaviour
    {
        [Range(1024, 65535)] public int port = 49168;
        public bool drawSkeletons = true;
        public string Status { get; private set; } = "Stopped";
        public ActorFrame[] Actors { get; private set; } = Array.Empty<ActorFrame>();
        public bool IsReceiving => connected && Time.realtimeSinceStartup - lastReceived < 2f;
        const int Limit = 8 * 1024 * 1024;
        Thread worker;
        volatile bool stopping, connected;
        volatile string pending, networkStatus;
        TcpClient client;
        float lastReceived;
        volatile int heartbeat;
        int seenHeartbeat;

        void OnEnable()
        {
            stopping = false;
            int selectedPort = port;
            worker = new Thread(() => Receive(selectedPort)) { IsBackground = true, Name = "RGB MoCap Unity" };
            worker.Start();
        }
        void OnDisable()
        {
            stopping = true;
            try { client?.Close(); } catch (ObjectDisposedException) { }
            worker?.Join(1500);
            worker = null;
            connected = false;
            pending = null;
            Actors = Array.Empty<ActorFrame>();
            Status = "Stopped";
        }
        static void ReadExact(Stream stream, byte[] buffer)
        {
            int offset = 0;
            while (offset < buffer.Length)
            {
                int count = stream.Read(buffer, offset, buffer.Length - offset);
                if (count == 0) throw new EndOfStreamException();
                offset += count;
            }
        }
        void Receive(int selectedPort)
        {
            while (!stopping)
            {
                try
                {
                    networkStatus = "Connecting to RGB MoCap...";
                    using (var connection = new TcpClient())
                    {
                        client = connection;
                        connection.NoDelay = true;
                        connection.ReceiveTimeout = 2000;
                        // Loopback only, matching the application's local stream listener.
                        connection.Connect("127.0.0.1", selectedPort);
                        if (stopping) break;
                        connected = true;
                        networkStatus = "Connected";
                        using (var stream = connection.GetStream())
                        {
                            var header = new byte[4];
                            while (!stopping)
                            {
                                ReadExact(stream, header);
                                int length = header[0] | header[1] << 8 | header[2] << 16 | header[3] << 24;
                                if (length < 0 || length > Limit) throw new IOException("Invalid packet length");
                                if (length > 0)
                                {
                                    var bytes = new byte[length];
                                    ReadExact(stream, bytes);
                                    // A single replaceable slot prevents playback latency accumulating.
                                    Interlocked.Exchange(ref pending, Encoding.UTF8.GetString(bytes));
                                }
                                Interlocked.Increment(ref heartbeat);
                            }
                        }
                    }
                }
                catch (Exception error) when (error is IOException || error is SocketException || error is ObjectDisposedException)
                { networkStatus = "Waiting for RGB MoCap: " + error.Message; }
                finally { connected = false; client = null; }
                for (int i = 0; i < 10 && !stopping; i++) Thread.Sleep(100);
            }
        }
        public static bool Valid(StreamFrame value)
        {
            if (value == null || value.version != 1 || value.actors == null || value.actors.Length > 128) return false;
            var ids = new HashSet<string>();
            foreach (var a in value.actors)
            {
                if (a == null || String.IsNullOrEmpty(a.id) || !ids.Add(a.id) || String.IsNullOrEmpty(a.rig) || a.names == null) return false;
                int count = a.names.Length;
                if (count < 1 || count > 256 || a.parents == null || a.parents.Length != count ||
                    a.rest == null || a.pose == null || a.rest.Length != count*12 || a.pose.Length != count*12) return false;
                for (int i = 0; i < count; i++)
                {
                    if (String.IsNullOrEmpty(a.names[i]) || a.parents[i] < -1 || a.parents[i] >= i) return false;
                    foreach (var data in new[] {a.rest, a.pose})
                    {
                        for (int j = i*12; j < i*12+12; j++) if (float.IsNaN(data[j]) || float.IsInfinity(data[j])) return false;
                        int k = i*12;
                        Vector3 x = new Vector3(data[k], data[k+4], data[k+8]);
                        Vector3 y = new Vector3(data[k+1], data[k+5], data[k+9]);
                        Vector3 z = new Vector3(data[k+2], data[k+6], data[k+10]);
                        if (Mathf.Abs(x.sqrMagnitude-1) > .02f || Mathf.Abs(y.sqrMagnitude-1) > .02f ||
                            Mathf.Abs(z.sqrMagnitude-1) > .02f || Vector3.Dot(Vector3.Cross(x,y),z) < .98f) return false;
                    }
                }
            }
            return true;
        }
        void Update()
        {
            Status = networkStatus ?? "Connecting...";
            int beat = heartbeat;
            if (beat != seenHeartbeat) { seenHeartbeat = beat; lastReceived = Time.realtimeSinceStartup; }
            string raw = Interlocked.Exchange(ref pending, null);
            if (raw == null) return;
            try
            {
                var value = JsonUtility.FromJson<StreamFrame>(raw);
                if (!Valid(value)) throw new ArgumentException("Invalid skeleton packet");
                Actors = value.actors;
            }
            catch (ArgumentException error) { Status = error.Message; Actors = Array.Empty<ActorFrame>(); }
        }
        void OnDrawGizmos()
        {
            if (!drawSkeletons || !IsReceiving) return;
            for (int a = 0; a < Actors.Length; a++)
            {
                Gizmos.color = Color.HSVToRGB((a * .23f) % 1f, .8f, 1f);
                var actor = Actors[a];
                for (int i = 0; i < actor.parents.Length; i++)
                    if (actor.parents[i] >= 0)
                        Gizmos.DrawLine(transform.TransformPoint(actor.Position(actor.pose, i)),
                                        transform.TransformPoint(actor.Position(actor.pose, actor.parents[i])));
            }
        }
    }
}
