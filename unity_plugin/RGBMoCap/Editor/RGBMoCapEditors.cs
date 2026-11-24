using UnityEditor;
using UnityEngine;

namespace RGBMoCap.Editor
{
    public static class CreateReceiver
    {
        [MenuItem("GameObject/RGB MoCap/Stream Receiver", false, 10)]
        public static void Create()
        {
            var go = new GameObject("RGB MoCap Receiver", typeof(RGBMoCapReceiver));
            Undo.RegisterCreatedObjectUndo(go, "Create RGB MoCap receiver");
            Selection.activeGameObject=go;
        }
    }
    [CustomEditor(typeof(RGBMoCapReceiver))]
    public sealed class ReceiverEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var receiver=(RGBMoCapReceiver)target;
            EditorGUILayout.HelpBox("In RGB MoCap: Stream > Unity, check people, then play or scrub. Enter Play Mode here. One receiver serves all characters on this PC.", MessageType.Info);
            EditorGUILayout.LabelField(receiver.Status);
            foreach(var a in receiver.Actors) EditorGUILayout.LabelField(a.name, "Frame "+a.frame+" · "+a.names.Length+" bones");
            if(Application.isPlaying) Repaint();
        }
    }
    [CustomEditor(typeof(RGBMoCapActor))]
    public sealed class ActorEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var actor=(RGBMoCapActor)target;
            EditorGUILayout.HelpBox("Start in a reference pose matching the source (normally T-pose), facing +Z. Animator is disabled while this component runs. Body/fingers use rotations; only pelvis translation is applied. Stop Play Mode to restore the original pose.", MessageType.Info);
            if(actor.receiver!=null && actor.receiver.Actors.Length>0)
            {
                var actors=actor.receiver.Actors;
                var labels=new string[actors.Length+1]; labels[0]="Select streaming person...";
                int current=0;
                for(int i=0;i<actors.Length;i++)
                { labels[i+1]=actors[i].name+" ("+actors[i].id.Substring(0,Mathf.Min(8,actors[i].id.Length))+")"; if(actor.actorIdOrName==actors[i].id) current=i+1; }
                int chosen=EditorGUILayout.Popup("Person",current,labels);
                if(chosen>0 && chosen!=current)
                { Undo.RecordObject(actor,"Select streaming actor"); actor.actorIdOrName=actors[chosen-1].id; actor.Rebind(); EditorUtility.SetDirty(actor); }
            }
            if(GUILayout.Button("Rebuild bone mapping")) actor.Rebind();
            EditorGUILayout.LabelField(actor.Status);
            if(Application.isPlaying) Repaint();
        }
    }
}
