// Windows Explorer's native folder picker (IFileDialog/FOS_PICKFOLDERS).
using System;
using System.Runtime.InteropServices;

static class FolderPicker {
    [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")] class FileOpenDialog { }
    [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IFileDialog {
        [PreserveSig] int Show(IntPtr owner);
        void SetFileTypes(uint count, IntPtr types);
        void SetFileTypeIndex(uint index); void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie); void Unadvise(uint cookie);
        void SetOptions(uint options); void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem item); void SetFolder(IShellItem item);
        void GetFolder(out IShellItem item); void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName(out IntPtr name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, int placement);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int result); void SetClientGuid(ref Guid guid); void ClearClientData(); void SetFilter(IntPtr filter);
    }
    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IShellItem {
        void BindToHandler(IntPtr context, ref Guid handler, ref Guid iid, out IntPtr result);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint kind, out IntPtr name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem item, uint hint, out int order);
    }
    [DllImport("shell32.dll", CharSet=CharSet.Unicode, PreserveSig=false)]
    static extern void SHCreateItemFromParsingName(string path, IntPtr context, ref Guid iid, out IShellItem item);
    public static string Pick(IntPtr owner, string initial) {
        var dialog=(IFileDialog)new FileOpenDialog();
        IShellItem folder=null, result=null;
        try {
            uint options; dialog.GetOptions(out options);
            dialog.SetOptions(options | 0x20 | 0x40 | 0x800 | 0x02000000);
            dialog.SetTitle("Choose RGB MoCap installation folder"); dialog.SetOkButtonLabel("Select Folder");
            if (System.IO.Directory.Exists(initial)) {
                Guid iid=typeof(IShellItem).GUID;
                SHCreateItemFromParsingName(initial,IntPtr.Zero,ref iid,out folder); dialog.SetFolder(folder);
            }
            int code=dialog.Show(owner);
            if (code==unchecked((int)0x800704C7)) return null;
            Marshal.ThrowExceptionForHR(code);
            dialog.GetResult(out result); IntPtr path; result.GetDisplayName(0x80058000,out path);
            try { return Marshal.PtrToStringUni(path); } finally { Marshal.FreeCoTaskMem(path); }
        } finally {
            if (result!=null) Marshal.ReleaseComObject(result);
            if (folder!=null) Marshal.ReleaseComObject(folder);
            Marshal.ReleaseComObject(dialog);
        }
    }
}
