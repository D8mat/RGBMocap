// Native Windows bootstrap UI. No system Python, global PATH or registry edits.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

class SetupForm : Form {
    readonly string setupFolder = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
    string root = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
    readonly JavaScriptSerializer json = new JavaScriptSerializer { MaxJsonLength = 8 * 1024 * 1024 };
    readonly Dictionary<string, object> meta;
    readonly Label folder = new Label(), sizes = new Label(), status = new Label(), elapsed = new Label();
    readonly ProgressBar progress = new ProgressBar();
    readonly TextBox details = new TextBox();
    readonly Button install = new Button(), cancel = new Button(), browse = new Button();
    readonly ToolTip tips = new ToolTip();
    readonly CheckBox showDetails = new CheckBox();
    readonly Stopwatch watch = new Stopwatch();
    readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
    bool busy, cancelling, committing, wantClose, complete;
    int generation;
    string lastError, subtaskEstimate = "";
    readonly object logLock = new object();
    Process current;
    string cancelFile;
    Mutex mutex;
    SafeJob job = new SafeJob();

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool GetDiskFreeSpaceEx(string path, out ulong available, out ulong total, out ulong free);
    static string GiB(double bytes) { return (bytes / 1073741824d).ToString("0.0") + " GiB"; }
    long Bytes(string key) { return Convert.ToInt64(meta[key]); }
    void UI(Action action) { if (!IsDisposed) { if (InvokeRequired) BeginInvoke(action); else action(); } }
    void Log(string line) {
        try { lock (logLock) { string d = Path.Combine(root, ".setup"); Directory.CreateDirectory(d); string path = Path.Combine(d, "setup.log"); if (File.Exists(path) && new FileInfo(path).Length > 4 * 1024 * 1024) { string old = path + ".previous"; if (File.Exists(old)) File.Delete(old); File.Move(path, old); } File.AppendAllText(path, line + Environment.NewLine, Encoding.UTF8); } } catch { }
        UI(delegate { if (details.TextLength > 80000) details.Text = details.Text.Substring(40000); details.AppendText(line + Environment.NewLine); });
    }

    public SetupForm() {
        using (var reader = new StreamReader(Assembly.GetExecutingAssembly().GetManifestResourceStream("setup.json")))
            meta = json.Deserialize<Dictionary<string, object>>(reader.ReadToEnd());
        Text = "RGB MoCap Setup"; Font = new Font("Segoe UI", 10); BackColor = Color.FromArgb(37, 42, 48); ForeColor = Color.Gainsboro;
        ClientSize = new Size(750, 510); MinimumSize = MaximumSize = Size; MaximizeBox = false; StartPosition = FormStartPosition.CenterScreen;
        var title = new Label { Text = "Install RGB MoCap", Font = new Font(Font.FontFamily, 19, FontStyle.Bold), Bounds = new Rectangle(28, 22, 680, 40) };
        var intro = new Label { Text = "Choose a folder for the app and all AI runtimes.\nModel files are prepared separately in Model Manager.", Bounds = new Rectangle(30, 72, 680, 52) };
        folder.Text = "Install location: " + root; folder.Bounds = new Rectangle(30, 137, 560, 30); folder.AutoEllipsis = true;
        tips.SetToolTip(folder, root);
        browse.Text = "Browse..."; browse.Bounds = new Rectangle(600, 132, 115, 35);
        browse.Click += delegate { try { string selected = FolderPicker.Pick(Handle, root); if (selected != null) SetLocation(selected); } catch (Exception e) { status.Text = e.Message; } };
        tips.SetToolTip(browse, "Choose an installation folder. By default, Setup installs beside this file.");
        sizes.Bounds = new Rectangle(30, 177, 685, 108);
        progress.Bounds = new Rectangle(30, 302, 685, 21); progress.Style = ProgressBarStyle.Continuous;
        status.Text = "Ready. Internet is required for the first installation."; status.Bounds = new Rectangle(30, 335, 685, 43);
        elapsed.Bounds = new Rectangle(30, 380, 685, 25);
        showDetails.Text = "Show details"; showDetails.Bounds = new Rectangle(30, 420, 150, 28);
        details.Multiline = true; details.ReadOnly = true; details.ScrollBars = ScrollBars.Vertical; details.BackColor = Color.FromArgb(25, 29, 34); details.ForeColor = Color.Gainsboro;
        details.Bounds = new Rectangle(30, 460, 685, 180); details.Visible = false; details.Font = new Font("Consolas", 9);
        install.Text = "Install All Runtimes"; install.Bounds = new Rectangle(362, 421, 215, 37);
        cancel.Text = "Close"; cancel.Bounds = new Rectangle(592, 421, 123, 37);
        Controls.AddRange(new Control[] { title, intro, folder, browse, sizes, progress, status, elapsed, showDetails, details, install, cancel });
        showDetails.CheckedChanged += delegate { details.Visible = showDetails.Checked; MinimumSize = MaximumSize = System.Drawing.Size.Empty; ClientSize = new Size(750, showDetails.Checked ? 660 : 510); MinimumSize = MaximumSize = Size; };
        cancel.Click += delegate { if (busy) RequestCancel(); else Close(); };
        install.Click += async delegate { if (complete) { Process.Start(new ProcessStartInfo(Path.Combine(root, "RGBMoCap.exe")) { WorkingDirectory = root, UseShellExecute = true }); Close(); } else await StartInstall(); };
        FormClosing += delegate(object sender, FormClosingEventArgs e) { if (busy) { e.Cancel = true; wantClose = true; RequestCancel(); } };
        timer.Interval = 1000; timer.Tick += delegate { if (busy) elapsed.Text = "Elapsed: " + watch.Elapsed.ToString(@"hh\:mm\:ss") + subtaskEstimate; }; timer.Start();
        RefreshSpace();
    }

    public void SetLocation(string path) {
        if (busy) throw new InvalidOperationException("Installation is running.");
        root = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar);
        if (root.EndsWith(":")) root += Path.DirectorySeparatorChar;
        complete = false; install.Text = "Install All Runtimes";
        folder.Text = "Install location: " + root; tips.SetToolTip(folder, root);
        status.Text = "Ready. Internet is required for the first installation.";
        RefreshSpace();
    }

    void RefreshSpace() {
        ulong available, total, free;
        string disk = GetDiskFreeSpaceEx(Directory.Exists(root) ? root : Path.GetPathRoot(root), out available, out total, out free) ? GiB(available) : "Unavailable";
        sizes.Text = "Download (estimated): " + GiB(Bytes("download_bytes")) + "\n" +
            "Installed app + runtimes (estimated): " + GiB(Bytes("installed_bytes_estimate")) + "\n" +
            "Free space required during first installation: " + GiB(Bytes("fresh_free_bytes_required")) + "\n" +
            "Available on this drive: " + disk;
    }

    static bool IsLink(string path) { return (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0; }
    void CheckFolder() {
        string parent = root;
        while (parent != null) {
            if ((Directory.Exists(parent) || File.Exists(parent)) && IsLink(parent)) throw new IOException("Choose a regular folder, not a linked folder.");
            parent = Path.GetDirectoryName(parent);
        }
        Directory.CreateDirectory(root);
        string setup = Path.Combine(root, ".setup");
        if (Directory.Exists(setup) && IsLink(setup)) throw new IOException("The setup working folder must not be a link.");
        if (!File.Exists(Path.Combine(root, "BUILD_INFO.json")) && !File.Exists(Path.Combine(root, "SETUP_RECEIPT.json")))
            foreach (string name in new[] { "mocap", "_internal", "third_party", "runtime", "runtime_exmr", "runtime_flow", "runtime_hand", "runtime_htd" })
                if (Directory.Exists(Path.Combine(root, name))) throw new IOException("Choose a dedicated folder. This folder contains unrelated app files.");
        ulong available, total, free;
        if (!GetDiskFreeSpaceEx(root, out available, out total, out free)) throw new IOException("Could not check available disk space.");
        if (!File.Exists(Path.Combine(root, "SETUP_RECEIPT.json")) && !Directory.Exists(setup) && available < (ulong)Bytes("fresh_free_bytes_required")) throw new IOException("Not enough free space. Setup needs " + GiB(Bytes("fresh_free_bytes_required")) + " during installation.");
        foreach (var p in Process.GetProcessesByName("RGBMoCap")) {
            try { if (String.Equals(p.MainModule.FileName, Path.Combine(root, "RGBMoCap.exe"), StringComparison.OrdinalIgnoreCase)) throw new IOException("Close RGB MoCap in this folder, then retry Setup."); }
            catch (System.ComponentModel.Win32Exception) { }
        }
        string probe = Path.Combine(root, ".rgbmocap-write-" + Guid.NewGuid().ToString("N"));
        using (var file = File.Create(probe)) { } File.Delete(probe);
    }

    public Dictionary<string, object> Inspection() {
        return new Dictionary<string, object> { { "root", root }, { "default_root", setupFolder }, { "folder_selection", true }, { "download_bytes", Bytes("download_bytes") },
            { "installed_bytes_estimate", Bytes("installed_bytes_estimate") }, { "fresh_free_bytes_required", Bytes("fresh_free_bytes_required") },
            { "all_runtimes", true }, { "models_included", false }, { "version", meta["version"] } };
    }

    string ExtractPayload() {
        string output = Path.Combine(root, ".setup", "bootstrap"); EnsureNoLinks(output); Directory.CreateDirectory(output);
        string archive = Path.Combine(output, "payload.zip"); string partial = archive + ".partial";
        using (var exe = File.OpenRead(Assembly.GetExecutingAssembly().Location)) {
            if (exe.Length < 56) throw new IOException("Setup is incomplete.");
            exe.Position = exe.Length - 56; var reader = new BinaryReader(exe); long length = reader.ReadInt64(); byte[] expected = reader.ReadBytes(32);
            if (Encoding.ASCII.GetString(reader.ReadBytes(16)) != "RGBMOCAPSETUPV1!" || length <= 0 || length > exe.Length - 56) throw new IOException("Setup payload is missing.");
            exe.Position = exe.Length - 56 - length;
            using (var digest = SHA256.Create()) using (var stream = File.Create(partial)) {
                var buffer = new byte[4 * 1024 * 1024]; long left = length;
                while (left > 0) {
                    if (cancelling) throw new OperationCanceledException();
                    int n = exe.Read(buffer, 0, (int)Math.Min(left, buffer.Length)); if (n == 0) throw new IOException("Incomplete setup file.");
                    stream.Write(buffer, 0, n); digest.TransformBlock(buffer, 0, n, buffer, 0); left -= n;
                    int value = (int)((length - left) * 100 / length); UI(delegate { progress.Style = ProgressBarStyle.Continuous; progress.Value = value; status.Text = "Checking Setup files..."; });
                }
                digest.TransformFinalBlock(new byte[0], 0, 0); if (!digest.Hash.SequenceEqual(expected)) throw new IOException("Setup file verification failed. Download Setup again.");
            }
        }
        if (File.Exists(archive)) File.Delete(archive); File.Move(partial, archive);
        using (var zip = ZipFile.OpenRead(archive)) {
            string prefix = Path.GetFullPath(output) + Path.DirectorySeparatorChar;
            foreach (var entry in zip.Entries) {
                if (cancelling) throw new OperationCanceledException();
                string path = Path.GetFullPath(Path.Combine(output, entry.FullName.Replace('/', Path.DirectorySeparatorChar)));
                if (!path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || entry.FullName.Contains(":")) throw new IOException("Invalid setup path.");
                EnsureNoLinks(path);
                if (entry.FullName.EndsWith("/")) { Directory.CreateDirectory(path); continue; }
                Directory.CreateDirectory(Path.GetDirectoryName(path)); entry.ExtractToFile(path, true);
            }
        }
        File.Delete(archive); return output;
    }

    void EnsureNoLinks(string path) {
        string p = Path.GetFullPath(path);
        while (!String.Equals(p, root, StringComparison.OrdinalIgnoreCase)) {
            if ((Directory.Exists(p) || File.Exists(p)) && IsLink(p)) throw new IOException("An installer path is a link: " + p);
            p = Path.GetDirectoryName(p); if (p == null) throw new IOException("Invalid installer path.");
        }
    }

    static string Quote(string value) {
        var b = new StringBuilder("\""); int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { b.Append('\\', slashes * 2 + 1).Append('"'); slashes = 0; continue; }
            b.Append('\\', slashes).Append(c); slashes = 0;
        }
        return b.Append('\\', slashes * 2).Append('"').ToString();
    }

    void OnLine(string line) {
        if (String.IsNullOrEmpty(line)) return;
        Log(line);
        if (!line.StartsWith("@@SETUP@@")) return;
        try {
            var data = json.Deserialize<Dictionary<string, object>>(line.Substring(9));
            if (data.ContainsKey("type") && Convert.ToString(data["type"]) == "error") lastError = Convert.ToString(data["message"]);
            UI(delegate {
                if (data.ContainsKey("message")) status.Text = Convert.ToString(data["message"]);
                subtaskEstimate = data.ContainsKey("remaining_seconds") ? "   Download remaining (approx.): " + TimeSpan.FromSeconds(Math.Min(86400, Convert.ToDouble(data["remaining_seconds"]))).ToString(@"hh\:mm\:ss") : "";
                if (data.ContainsKey("committing") && Convert.ToBoolean(data["committing"])) { committing = true; cancel.Enabled = false; }
                long all = data.ContainsKey("total") ? Convert.ToInt64(data["total"]) : 0;
                long now = data.ContainsKey("current") ? Convert.ToInt64(data["current"]) : 0;
                if (all > 0) { progress.Style = ProgressBarStyle.Continuous; progress.Value = (int)Math.Max(0, Math.Min(100, now * 100 / all)); }
                else progress.Style = ProgressBarStyle.Marquee;
            });
        } catch (Exception e) { Log("Status: " + e.Message); }
    }

    void Run(string exe, params string[] args) {
        if (cancelling) throw new OperationCanceledException();
        var si = new ProcessStartInfo(exe, String.Join(" ", args.Select(Quote))) { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
        foreach (string name in si.EnvironmentVariables.Keys.Cast<string>().ToArray()) if (name.StartsWith("PYTHON", StringComparison.OrdinalIgnoreCase) || name.StartsWith("PIP_", StringComparison.OrdinalIgnoreCase) || name.StartsWith("UV_", StringComparison.OrdinalIgnoreCase)) si.EnvironmentVariables.Remove(name);
        string setup = Path.Combine(root, ".setup"); string temp = Path.Combine(setup, "tmp"); Directory.CreateDirectory(temp);
        si.EnvironmentVariables["PYTHONUTF8"] = "1"; si.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"; si.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
        si.EnvironmentVariables["UV_PYTHON_INSTALL_DIR"] = Path.Combine(setup, "python"); si.EnvironmentVariables["UV_CACHE_DIR"] = Path.Combine(setup, "uv-cache");
        si.EnvironmentVariables["UV_NO_CONFIG"] = "1"; si.EnvironmentVariables["UV_PYTHON_PREFERENCE"] = "only-managed"; si.EnvironmentVariables["TEMP"] = si.EnvironmentVariables["TMP"] = temp;
        using (var process = new Process { StartInfo = si }) {
            current = process; process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { OnLine(e.Data); }; process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { OnLine(e.Data); };
            process.Start(); job.Add(process); process.BeginOutputReadLine(); process.BeginErrorReadLine(); process.WaitForExit(); current = null;
            if (cancelling && process.ExitCode != 0) throw new OperationCanceledException();
            if (process.ExitCode != 0) throw new IOException("Setup could not finish this step. Check the details and retry. Exit code: " + process.ExitCode);
        }
    }

    void RequestCancel() {
        if (committing) return; cancelling = true; cancel.Enabled = false; status.Text = "Stopping safely. Verified downloads will be kept for retry...";
        if (cancelFile != null) try { File.WriteAllText(cancelFile, "cancel"); } catch { }
        int run = generation; SafeJob ownedJob = job;
        Process active = current;
        if (active != null && active.StartInfo.FileName.EndsWith("uv.exe", StringComparison.OrdinalIgnoreCase)) try { active.Kill(); } catch { }
        Task.Run(async delegate { await Task.Delay(30000); if (busy && cancelling && generation == run) ownedJob.Kill(); });
    }

    void SafeDelete(string path) {
        string prefix = Path.GetFullPath(Path.Combine(root, ".setup")) + Path.DirectorySeparatorChar;
        string full = Path.GetFullPath(path); if (!full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) throw new IOException("Invalid cleanup path.");
        if (!Directory.Exists(full)) return;
        if (IsLink(full)) { Directory.Delete(full, false); return; }
        foreach (string file in Directory.EnumerateFiles(full)) { File.SetAttributes(file, FileAttributes.Normal); File.Delete(file); }
        foreach (string dir in Directory.EnumerateDirectories(full)) { if (IsLink(dir)) Directory.Delete(dir, false); else SafeDelete(dir); }
        Directory.Delete(full, false);
    }

    public async Task StartInstall() {
        if (busy) return;
        try { CheckFolder(); }
        catch (Exception e) { status.Text = e.Message; return; }
        byte[] key; using (var h = SHA256.Create()) key = h.ComputeHash(Encoding.UTF8.GetBytes(root.ToUpperInvariant()));
        bool created; mutex = new Mutex(true, "Local\\RGBMoCapSetup-" + BitConverter.ToString(key).Replace("-", ""), out created);
        if (!created) { mutex.Dispose(); mutex = null; status.Text = "Another Setup is already running in this folder."; return; }
        job.Dispose(); job = new SafeJob(); generation++; lastError = null;
        busy = true; browse.Enabled = false; cancelling = committing = false; install.Enabled = false; cancel.Enabled = true; cancel.Text = "Cancel"; progress.Style = ProgressBarStyle.Marquee; watch.Restart();
        try {
            await Task.Run(delegate {
                string resources = ExtractPayload(); string setup = Path.Combine(root, ".setup"); cancelFile = Path.Combine(setup, "cancel.request"); if (File.Exists(cancelFile)) File.Delete(cancelFile);
                UI(delegate { status.Text = "Preparing private Python. Please wait..."; progress.Style = ProgressBarStyle.Marquee; });
                string uv = Path.Combine(resources, "uv.exe"); string pythonDir = Path.Combine(setup, "python");
                string py = Path.Combine(root, "runtime", "python.exe");
                bool useInstalled = File.Exists(py) && File.Exists(Path.Combine(root, "SETUP_RECEIPT.json"));
                if (useInstalled) {
                    // A healthy installation can be checked again without network access.
                    try { Run(py, "-B", "-c", "import sys; assert sys.version_info[:3] == (3,10,22)"); }
                    catch (OperationCanceledException) { throw; }
                    catch { useInstalled = false; }
                }
                if (!useInstalled) {
                    Run(uv, "python", "install", "3.10.22", "--install-dir", pythonDir, "--no-bin", "--no-registry", "--no-config", "--no-progress", "--cache-dir", Path.Combine(setup, "uv-cache"));
                    py = Path.Combine(pythonDir, "cpython-3.10.22-windows-x86_64-none", "python.exe");
                }
                Run(py, "-B", Path.Combine(resources, "engine.py"), "--root", root, "--resources", resources, "--cancel", cancelFile);
                UI(delegate { status.Text = "Clearing temporary installation files..."; });
                foreach (string name in new[] { "downloads", "staging", "backups", "python", "tmp", "uv-cache" }) try { SafeDelete(Path.Combine(setup, name)); } catch (Exception e) { Log("Temporary cleanup: " + e.Message); }
                // Retain only the small repair scripts, installer helper and receipts.
                try { SafeDelete(Path.Combine(resources, "app")); } catch (Exception e) { Log(e.Message); }
            });
            complete = true; progress.Style = ProgressBarStyle.Continuous; progress.Value = 100; status.Text = "Installed and verified. Prepare model files in Model Manager."; install.Text = "Open RGB MoCap";
        } catch (OperationCanceledException) { status.Text = "Cancelled. Run Setup again to continue."; progress.Style = ProgressBarStyle.Continuous; }
          catch (Exception e) { status.Text = lastError ?? e.Message; Log(e.ToString()); showDetails.Checked = true; progress.Style = ProgressBarStyle.Continuous; }
        finally {
            busy = false; watch.Stop(); elapsed.Text = "Elapsed: " + watch.Elapsed.ToString(@"hh\:mm\:ss"); install.Enabled = cancel.Enabled = browse.Enabled = true; cancel.Text = "Close"; committing = false;
            if (mutex != null) { mutex.ReleaseMutex(); mutex.Dispose(); mutex = null; } RefreshSpace();
            if (wantClose) Close();
        }
    }

    protected override void Dispose(bool disposing) { if (disposing) { timer.Dispose(); tips.Dispose(); job.Dispose(); } base.Dispose(disposing); }

    [STAThread] static void Main(string[] args) {
        AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling", false); AppContext.SetSwitch("Switch.System.IO.BlockLongPaths", false);
        Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
        using (var form = new SetupForm()) {
            int target = Array.IndexOf(args, "--target");
            if (target >= 0 && target + 1 < args.Length) form.SetLocation(args[target + 1]);
            int at = Array.IndexOf(args, "--inspect");
            if (at >= 0 && at + 1 < args.Length) { File.WriteAllText(args[at + 1], form.json.Serialize(form.Inspection())); return; }
            int smoke = Array.IndexOf(args, "--smoke");
            if (smoke >= 0 && smoke + 1 < args.Length) {
                var t = new System.Windows.Forms.Timer { Interval = 900 }; t.Tick += delegate { t.Stop(); using (var b = new Bitmap(form.Width, form.Height)) { form.DrawToBitmap(b, new Rectangle(0, 0, b.Width, b.Height)); b.Save(args[smoke + 1] + ".png"); } File.WriteAllText(args[smoke + 1] + ".json", form.json.Serialize(form.Inspection())); form.Close(); }; t.Start();
            }
            if (args.Contains("--install")) form.Shown += async delegate { await form.StartInstall(); Environment.ExitCode = form.complete ? 0 : 1; form.Close(); };
            Application.Run(form);
        }
    }
}

sealed class SafeJob : IDisposable {
    IntPtr handle;
    [StructLayout(LayoutKind.Sequential)] struct Basic { public long PerProcess, PerJob; public uint Flags; public UIntPtr Min, Max; public uint Active; public UIntPtr Affinity; public uint Priority, Scheduling; }
    [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A, B, C, D, E, F; }
    [StructLayout(LayoutKind.Sequential)] struct Extended { public Basic Basic; public Io Io; public UIntPtr A, B, C, D; }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int kind, ref Extended value, uint length);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool TerminateJobObject(IntPtr job, uint code);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    public SafeJob() { handle = CreateJobObject(IntPtr.Zero, null); var e = new Extended(); e.Basic.Flags = 0x2000; SetInformationJobObject(handle, 9, ref e, (uint)Marshal.SizeOf(e)); }
    public void Add(Process process) { if (handle != IntPtr.Zero) AssignProcessToJobObject(handle, process.Handle); }
    public void Kill() { if (handle != IntPtr.Zero) TerminateJobObject(handle, 2); }
    public void Dispose() { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
}
