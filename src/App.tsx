import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, ChevronRight, CircleHelp, Download, Hand, Maximize, Minus, Plus, RotateCcw, Upload, Waves, X } from "lucide-react";
import { EffectRenderer } from "./effects/EffectRenderer";
import { gestureLabel, classifyHands, GestureStabilizer, type TrackedHand } from "./vision/gestures";
import { HandTracker } from "./vision/HandTracker";

type EffectId = "cursor" | "energy" | "draw" | "multi" | "portal" | "wireframe";
type CameraStatus = "initializing" | "camera-starting" | "ready" | "denied" | "unavailable" | "unsupported" | "loading-model" | "model-error";

const effects: { id: EffectId; title: string; caption: string }[] = [
  { id: "cursor", title: "Cursor Effects", caption: "Index-led pointer" },
  { id: "portal", title: "Hand Portal Filter", caption: "Live filters in a hand frame" },
  { id: "energy", title: "Energy Orb", caption: "Open palms shape the field" },
  { id: "draw", title: "Neon Air Drawing", caption: "Index-led persistent strokes" },
  { id: "multi", title: "Multi-Hand Lines", caption: "Two independent trails" },
  { id: "wireframe", title: "Neon Wireframe Hands", caption: "Linked palm crystals" },
];

const instructions: Record<EffectId, string[]> = {
  cursor: ["Raise your index finger.", "Curl the other fingers and move to guide the cursor."],
  portal: ["Place both hands apart to frame a four-corner portal.", "Hold an open hand for 1s, or press N, to cycle filters."],
  energy: ["Show two open hands.", "Move your hands apart or together to resize the orb."],
  draw: ["Raise only your index finger to draw.", "Pinch or curl to lift the pen; show an open hand for 1s to clear."],
  multi: ["Show both hands at once.", "Raise each index finger to draw separate lines."],
  wireframe: ["Show one or both hands to form palm crystals.", "Connect fingertips with beams; G/R/B/P adjust effects."],
};

function statusMessage(status: CameraStatus) {
  if (status === "initializing") return "Initializing AI vision";
  if (status === "camera-starting") return "Initializing camera";
  if (status === "loading-model") return "Loading vision model";
  if (status === "denied") return "Camera permission blocked";
  if (status === "unavailable") return "Camera unavailable";
  if (status === "unsupported") return "Browser camera support required";
  if (status === "model-error") return "Vision model failed to load";
  return "Camera & tracking online";
}

function errorHelp(status: CameraStatus) {
  if (status === "denied") return "Allow camera access in your browser settings, then reload.";
  if (status === "model-error") return "Check your internet connection and reload to retry the vision model.";
  if (status === "unsupported") return "Open this tool in a modern browser over localhost or HTTPS.";
  return "Check your camera connection and reload the page.";
}

function downloadPreset(effect: EffectId) {
  const data = { format: "gesture-studio-preset", version: 1, effect, exportedAt: new Date().toISOString() };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${effect}-preset.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const handsRef = useRef<TrackedHand[]>([]);
  const effectRef = useRef<EffectId>("cursor");
  const rendererRef = useRef<EffectRenderer | null>(null);
  const [selected, setSelected] = useState<EffectId>("cursor");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("initializing");
  const [handCount, setHandCount] = useState(0);
  const [gesture, setGesture] = useState("Initializing vision");
  const [fps, setFps] = useState(0);
  const [preview, setPreview] = useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);

  const chooseEffect = useCallback((effect: EffectId) => {
    setSelected(effect);
    effectRef.current = effect;
    setGesture(gestureLabel(effect, handsRef.current));
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    let stream: MediaStream | null = null;
    let tracker: HandTracker | null = null;
    let stopped = false;
    let animationFrame = 0;
    let lastInference = 0;
    let lastUi = 0;
    let thumbsUpStarted = 0;
    let thumbsUpLatched = false;
    let frames = 0;
    let fpsStart = performance.now();
    const gestureStabilizer = new GestureStabilizer();
    const renderer = new EffectRenderer(canvas);
    rendererRef.current = renderer;

    const loop = (time: number) => {
      if (stopped) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      renderer.resize(rect.width, rect.height, ratio);
      if (tracker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && time - lastInference >= 30) {
        lastInference = time;
        try {
          const result = tracker.detect(video);
          if (result) handsRef.current = gestureStabilizer.update(classifyHands(result));
        } catch (error) {
          console.error("Hand tracking frame failed.", error);
          setCameraStatus("model-error");
          tracker.close();
          tracker = null;
          stream?.getTracks().forEach((track) => track.stop());
          video.srcObject = null;
        }
        const thumbsUp = handsRef.current.some((hand) => hand.thumbsUp);
        if (thumbsUp) {
          if (!thumbsUpStarted) thumbsUpStarted = time;
          if (time - thumbsUpStarted > 850 && !thumbsUpLatched) {
            const currentIndex = effects.findIndex((effect) => effect.id === effectRef.current);
            chooseEffect(effects[(currentIndex + 1) % effects.length].id);
            thumbsUpLatched = true;
          }
        } else {
          thumbsUpStarted = 0;
          thumbsUpLatched = false;
        }
      }
      const size = { width: video.videoWidth || rect.width, height: video.videoHeight || rect.height };
      renderer.render(effectRef.current, handsRef.current, rect.width, rect.height, time, size, video);
      frames++;
      if (time - lastUi > 250) {
        setHandCount(handsRef.current.length);
        setGesture(gestureLabel(effectRef.current, handsRef.current));
        lastUi = time;
      }
      if (time - fpsStart >= 1000) {
        setFps(Math.round((frames * 1000) / (time - fpsStart)));
        frames = 0;
        fpsStart = time;
      }
      animationFrame = requestAnimationFrame(loop);
    };

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          setCameraStatus("unsupported");
          setGesture("Use a secure browser context");
          setLoading(false);
          return;
        }
        setCameraStatus("initializing");
        setGesture("Initializing AI vision");
        await new Promise<void>((resolve) => window.setTimeout(resolve, 160));
        if (stopped) return;
        setCameraStatus("loading-model");
        setGesture("Loading hand tracking model");
        tracker = new HandTracker();
        try {
          await tracker.initialize();
        } catch (error) {
          if (stopped) return;
          console.error("Unable to load the hand tracking model.", error);
          tracker.close();
          tracker = null;
          setCameraStatus("model-error");
          setGesture("Could not load hand tracking model");
          setLoading(false);
          return;
        }
        if (stopped) {
          tracker.close();
          return;
        }
        setCameraStatus("camera-starting");
        setGesture("Initializing camera");
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
          });
        } catch (error) {
          if (stopped) return;
          tracker?.close();
          tracker = null;
          const name = error instanceof Error ? error.name : "";
          if (name === "NotAllowedError" || name === "PermissionDeniedError") {
            setCameraStatus("denied");
            setGesture("Allow camera access to begin");
          } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
            setCameraStatus("unavailable");
            setGesture("No camera found");
          } else {
            setCameraStatus("unavailable");
            setGesture("Could not open camera");
          }
          setLoading(false);
          return;
        }
        if (stopped) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
        if (stopped) {
          tracker.close();
          return;
        }
        setCameraStatus("ready");
        setLoading(false);
        setGesture("Tracking ready");
        animationFrame = requestAnimationFrame(loop);
      } catch (error) {
        if (stopped) return;
        setCameraStatus("unavailable");
        setGesture("Could not start camera");
        setLoading(false);
        tracker?.close();
        tracker = null;
        stream?.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
        console.error("Unable to start the camera preview.", error);
      }
    };

    void start();
    const resizeObserver = new ResizeObserver(() => {
      const bounds = canvas.getBoundingClientRect();
      renderer.resize(bounds.width, bounds.height, Math.min(window.devicePixelRatio || 1, 2));
    });
    resizeObserver.observe(canvas);
    return () => {
      stopped = true;
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      tracker?.close();
      stream?.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
      rendererRef.current = null;
      handsRef.current = [];
    };
  }, [chooseEffect]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      const key = event.key.toLowerCase();
      if (key === "m") {
        const currentIndex = effects.findIndex((effect) => effect.id === effectRef.current);
        chooseEffect(effects[(currentIndex + 1) % effects.length].id);
      } else if (key === "n") {
        rendererRef.current?.nextPortalFilter();
        setMessage(`Portal filter · ${rendererRef.current?.currentPortalFilter ?? ""}`);
        window.setTimeout(() => setMessage(""), 1800);
      } else if (key === "c") {
        rendererRef.current?.clearDrawing();
      } else if (key === "u") {
        rendererRef.current?.undoDrawing();
      } else if (key === "g") {
        rendererRef.current?.toggleSetting("glow");
      } else if (key === "r") {
        rendererRef.current?.toggleSetting("rotation");
      } else if (key === "b") {
        rendererRef.current?.toggleSetting("beams");
      } else if (key === "p") {
        rendererRef.current?.toggleSetting("particles");
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [chooseEffect]);

  const reset = () => rendererRef.current?.clearDrawing();
  const importPreset = async (file?: File) => {
    if (!file) return;
    try {
      let data: unknown;
      try {
        data = JSON.parse(await file.text());
      } catch {
        throw new Error("This file is not a supported Gesture Studio preset.");
      }
      if (typeof data !== "object" || data === null || !("format" in data) || !("effect" in data) ||
          data.format !== "gesture-studio-preset") {
        throw new Error("This file is not a supported Gesture Studio preset.");
      }
      const importedEffect = effects.find((effect) => effect.id === data.effect);
      if (!importedEffect) throw new Error("This preset names an unavailable effect.");
      chooseEffect(importedEffect.id);
      setMessage(`Preset loaded · ${importedEffect.title}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not read this preset.");
    } finally {
      window.setTimeout(() => setMessage(""), 3500);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const activeEffect = effects.find((effect) => effect.id === selected)!;
  const online = cameraStatus === "ready";

  return (
    <main className="studio">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Waves size={17} strokeWidth={1.7} /></div>
          <div><span className="brand-name">GESTURE</span><span className="brand-sub">STUDIO / 01</span></div>
        </div>

        <section className="effects-section" aria-label="Effects Library">
          <div className="section-label">EFFECTS LIBRARY <span>{String(effects.length).padStart(2, "0")}</span></div>
          <nav className="effect-list">
            {effects.map((effect, index) => (
              <button
                className={`effect-option ${selected === effect.id ? "selected" : ""}`}
                key={effect.id}
                onClick={() => chooseEffect(effect.id)}
                aria-pressed={selected === effect.id}
                aria-label={effect.title}
                title={effect.title}
              >
                <span className="effect-index">0{index + 1}</span>
                <span className="effect-name">{effect.title}<small>{effect.caption}</small></span>
                {selected === effect.id && <ChevronRight className="effect-arrow" size={15} />}
              </button>
            ))}
          </nav>
        </section>

        <section className="instructions">
          <div className="section-label">GESTURE</div>
          <div className="instruction-copy" key={selected}>
            {instructions[selected].map((line) => <p key={line}><Hand size={12} />{line}</p>)}
          </div>
        </section>

        <div className="sidebar-spacer" />

        <section className="status-block">
          <div className={`status-light ${online ? "online" : ""}`} />
          <div className="status-copy"><span>{cameraStatus === "ready" && handCount ? `${handCount} HAND${handCount > 1 ? "S" : ""} TRACKED` : cameraStatus === "ready" ? "AI HAND GESTURE SYSTEM ONLINE" : statusMessage(cameraStatus)}</span><small>{cameraStatus === "ready" ? gesture : "VISION SYSTEM"}</small></div>
          {loading && <span className="loading-ring" />}
        </section>

        <div className="library-actions">
          <button onClick={() => downloadPreset(selected)} title="Download this effect preset"><Download size={14} /><span>Export preset</span></button>
          <button onClick={() => fileRef.current?.click()} title="Import an effect preset"><Upload size={14} /><span>Import preset</span></button>
          <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importPreset(event.target.files?.[0])} />
        </div>
        <div className="sidebar-foot">LOCAL CAMERA · NO RECORDING</div>
      </aside>

      <section className="stage" aria-label={`${activeEffect.title} canvas`}>
        <video ref={videoRef} className={`camera-feed ${selected === "draw" ? (preview ? "pip-hidden" : "pip-preview") : preview ? "feed-dimmed" : "portal-base"}`} autoPlay muted playsInline />
        <canvas ref={canvasRef} className="effects-canvas" aria-label="Gesture effects canvas" />
        <div className="stage-vignette" />

        <header className="stage-top">
          <div className="active-effect"><span className="active-dot" /><span>{activeEffect.title}</span><span className="effect-separator">/</span><small>{gesture}</small></div>
          <div className="stage-tools">
            <span className={`camera-tag ${online ? "is-online" : ""}`}><span />{online ? "LIVE" : "STANDBY"}</span>
            <span className="fps-tag">{fps} <small>FPS</small></span>
          </div>
        </header>

        {loading && <div className="boot-screen"><div className="boot-orb"><Waves size={26} /></div><p>INITIALIZING AI VISION</p><span>{statusMessage(cameraStatus).toUpperCase()}</span><div className="boot-progress"><i /></div></div>}
        {!loading && !online && <div className="camera-error"><Camera size={22} /><p>{statusMessage(cameraStatus)}</p><small>{errorHelp(cameraStatus)}</small></div>}
        {!loading && online && selected === "draw" && <div className="canvas-hint">DRAW WITH YOUR INDEX FINGER</div>}

        <div className="stage-bottom">
          <div className="mode-chip"><span>MODE</span>{activeEffect.title.toUpperCase()}</div>
          <div className="stage-actions">
            {selected === "draw" && <button className="icon-button" onClick={reset} title="Clear drawing"><RotateCcw size={15} /></button>}
            <button className={`icon-button ${preview ? "is-toggled" : ""}`} onClick={() => setPreview(!preview)} title={selected === "draw" ? (preview ? "Show camera picture-in-picture" : "Hide camera picture-in-picture") : preview ? "Show camera feed" : "Dim camera feed"}><Camera size={15} /></button>
            <button className="icon-button" onClick={() => document.fullscreenElement ? void document.exitFullscreen() : void document.documentElement.requestFullscreen()} title="Toggle fullscreen"><Maximize size={15} /></button>
          </div>
        </div>
        {message && <div className="toast"><CircleHelp size={14} />{message}<button onClick={() => setMessage("")} aria-label="Dismiss"><X size={14} /></button></div>}
        {selected === "energy" && online && <div className="size-hint"><Minus size={12} />OPEN PALMS · HAND SPAN CONTROLS ORB<Plus size={12} /></div>}
        {selected === "portal" && online && <div className="size-hint">N / OPEN PALM · FILTER {rendererRef.current?.currentPortalFilter.toUpperCase()}</div>}
      </section>
    </main>
  );
}
