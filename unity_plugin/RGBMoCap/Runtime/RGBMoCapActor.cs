using System;
using System.Collections.Generic;
using UnityEngine;

namespace RGBMoCap
{
    [Serializable] public class BoneOverride { public string sourceBone; public Transform targetBone; }

    [DefaultExecutionOrder(11000)]
    [AddComponentMenu("RGB MoCap/Stream Actor")]
    public sealed class RGBMoCapActor : MonoBehaviour
    {
        public RGBMoCapReceiver receiver;
        [Tooltip("Persistent actor ID preferred. A unique person name is also accepted.")]
        public string actorIdOrName = "Person 1";
        public Animator animator;
        public bool rootMotion = true;
        [Min(.001f)] public float motionScale = 1f;
        [Tooltip("Optional per-bone overrides; also supports non-Humanoid rigs.")]
        public BoneOverride[] mapping = Array.Empty<BoneOverride>();
        public string Status { get; private set; } = "Waiting";
        class Binding { public int source; public Transform target; public Quaternion rest; public Quaternion sourceRest; }
        class Saved { public Transform bone; public Vector3 position; public Quaternion rotation; public Vector3 scale; public Quaternion world; }
        readonly List<Binding> bindings = new List<Binding>();
        readonly List<Saved> saved = new List<Saved>();
        string rigKey;
        int hipsIndex;
        Transform hips;
        Vector3 restHips;
        Quaternion heading;
        bool animatorWasEnabled;
        Animator capturedAnimator;

        void OnEnable()
        {
            if (animator == null) animator = GetComponentInChildren<Animator>();
            capturedAnimator = animator;
            if (capturedAnimator != null) { animatorWasEnabled = capturedAnimator.enabled; capturedAnimator.enabled = false; }
            CaptureReferencePose();
        }
        [ContextMenu("Capture matching reference pose")]
        public void CaptureReferencePose()
        {
            saved.Clear();
            foreach (var t in GetComponentsInChildren<Transform>(true))
                saved.Add(new Saved { bone=t, position=t.localPosition, rotation=t.localRotation, scale=t.localScale, world=t.rotation });
            heading = transform.rotation;
            rigKey = null;
        }
        void OnDisable()
        {
            foreach (var s in saved)
                if (s.bone != null) { s.bone.localPosition=s.position; s.bone.localRotation=s.rotation; s.bone.localScale=s.scale; }
            if (capturedAnimator != null) capturedAnimator.enabled = animatorWasEnabled;
            bindings.Clear(); rigKey = null;
        }
        public void Rebind() { rigKey = null; }
        static string Clean(string name)
        {
            int colon = name.LastIndexOf(':');
            if (colon >= 0) name = name.Substring(colon+1);
            return name.Replace("_", "").Replace(" ", "").ToLowerInvariant();
        }
        public static bool HumanBone(string name, out HumanBodyBones bone)
        {
            string key = Clean(name);
            foreach (HumanBodyBones candidate in Enum.GetValues(typeof(HumanBodyBones)))
                if (candidate != HumanBodyBones.LastBone && Clean(candidate.ToString()) == key) { bone=candidate; return true; }
            var aliases = new Dictionary<string, HumanBodyBones> {
                {"pelvis",HumanBodyBones.Hips}, {"spine1",HumanBodyBones.Chest}, {"spine2",HumanBodyBones.UpperChest},
                {"leftupleg",HumanBodyBones.LeftUpperLeg}, {"rightupleg",HumanBodyBones.RightUpperLeg},
                {"leftleg",HumanBodyBones.LeftLowerLeg}, {"rightleg",HumanBodyBones.RightLowerLeg},
                {"lefttoebase",HumanBodyBones.LeftToes}, {"righttoebase",HumanBodyBones.RightToes},
                {"leftarm",HumanBodyBones.LeftUpperArm}, {"rightarm",HumanBodyBones.RightUpperArm},
                {"leftforearm",HumanBodyBones.LeftLowerArm}, {"rightforearm",HumanBodyBones.RightLowerArm}
            };
            if (aliases.TryGetValue(key, out bone)) return true;
            foreach (string side in new[] {"Left", "Right"})
                foreach (string digit in new[] {"Thumb", "Index", "Middle", "Ring", "Pinky"})
                    for (int i=1; i<=3; i++)
                        if (key == Clean(side+"Hand"+digit+i))
                            return Enum.TryParse(side+(digit=="Pinky" ? "Little" : digit)+new[] {"Proximal","Intermediate","Distal"}[i-1], out bone);
            bone = HumanBodyBones.LastBone; return false;
        }
        Transform Target(string name)
        {
            foreach (var m in mapping) if (m != null && m.sourceBone == name) return m.targetBone;
            if (animator != null && animator.avatar != null && animator.avatar.isValid && animator.isHuman && HumanBone(name, out var human))
            { var t=animator.GetBoneTransform(human); if (t != null) return t; }
            Transform match=null;
            foreach (var s in saved)
                if (Clean(s.bone.name)==Clean(name)) { if (match!=null) return null; match=s.bone; }
            return match;
        }
        static int Depth(Transform t) { int n=0; while(t.parent!=null) { n++; t=t.parent; } return n; }
        void Bind(ActorFrame actor)
        {
            bindings.Clear(); hips=null; hipsIndex=-1;
            var used = new HashSet<Transform>();
            for (int i=0; i<actor.names.Length; i++)
            {
                Transform target=Target(actor.names[i]);
                if (target==null || !used.Add(target)) continue;
                var original=saved.Find(s=>s.bone==target);
                if (original==null) continue;
                bindings.Add(new Binding { source=i, target=target, rest=original.world, sourceRest=actor.Rotation(actor.rest,i) });
                if (HumanBone(actor.names[i], out var human) && human==HumanBodyBones.Hips)
                { hips=target; hipsIndex=i; }
            }
            bindings.Sort((a,b)=>Depth(a.target).CompareTo(Depth(b.target)));
            if (hips!=null)
            {
                // Restore saved local pose before measuring the reference pelvis position.
                foreach(var s in saved) { s.bone.localPosition=s.position; s.bone.localRotation=s.rotation; }
                restHips=hips.position;
            }
            rigKey=actor.id+actor.rig;
            Status = bindings.Count+" mapped bones"+(hips==null ? " — pelvis not mapped" : "");
        }
        void LateUpdate()
        {
            if (receiver==null || !receiver.IsReceiving) { Status="Waiting for stream (holding last pose)"; return; }
            ActorFrame selected=null;
            foreach(var a in receiver.Actors)
                if (a.id==actorIdOrName) { selected=a; break; }
            if(selected==null)
                foreach(var a in receiver.Actors)
                    if(a.name==actorIdOrName) { if(selected!=null) { Status="Duplicate name: select actor ID"; return; } selected=a; }
            if(selected==null) { Status="Person not active at this frame"; return; }
            if(rigKey!=selected.id+selected.rig) Bind(selected);
            Quaternion invHeading=Quaternion.Inverse(heading);
            foreach(var b in bindings)
                b.target.rotation=heading * selected.Rotation(selected.pose,b.source) * Quaternion.Inverse(b.sourceRest) * invHeading * b.rest;
            if(hips!=null)
                hips.position=restHips + (rootMotion ? heading * (selected.Position(selected.pose,hipsIndex)-selected.Position(selected.rest,hipsIndex)) * motionScale : Vector3.zero);
            Status=bindings.Count+" bones · frame "+selected.frame;
        }
    }
}
