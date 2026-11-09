# Song Fusion & Techno Aranžér

Lokální nástroj pro fúzi skladeb a produkční aranžování:
- **AI Režisér (Autopilot):** popíšeš česky svou představu → lokální LLM (`qwen3:30b-a3b` / `qwen3:8b`) se jednorázově načte na GPU, analyzuje obě skladby, vyplní celé rozhraní (tempo, tóninu, scénář, riffy, 303 acid, mix i ACE-Step), vysvětlí *co, jak a proč* nastavil a okamžitě se odpojí z GPU (`keep_alive=0`).
- **Živý monitoring procesů a GPU:** horní status panel sleduje v reálném čase vytížení GPU VRAM, stav modelů v Ollamě i ACE-Stepu a všechny spouštěné podprocesy (`ffmpeg`, `demucs`, `ollama`, `arranger`, `acestep`, `lora`) včetně PID a doby trvání.
- **DSP Aranžér:** staví dlouhé klubové aranže podle scénáře (`intro`, `build`, `drop`, `break`, `outro`), drží 100% pevný beat grid, natahuje stemy na cílové BPM (`rubberband`), přelaďuje podle Camelot kola, skládá riffy ze Songu B, syntetizuje TB-303 acid linku (včetně exportu `acid.mid`), přidává sidechain pumpu, snare roll, risery, impakty a exportuje 24-bit FLAC master, stemy i markery.
- **Generativní AI Fúze (ACE-Step 1.5 turbo):** 7 receptů (INT8 + CPU offload pro 6 GB VRAM).

## Spuštění
```bat
Fusion.bat          # UI na http://localhost:7861
```
Instalace od nuly: `powershell -ExecutionPolicy Bypass -File setup.ps1`

## Struktura modulů v `fusion/`
- `monitor.py` — centrální sledování běžících procesů (PID, časy), stavu GPU VRAM a odpojování modelů z GPU
- `director.py` — AI Režisér (Autopilot celého UI s JSON validací a automatickým GPU unloadem)
- `arranger.py` — DSP aranžér podle scénáře, cílové BPM a tónina, náhledy riffů/acidu, export stemů/MIDI/markerů
- `scenario.py` — parser scénáře, presety, odvození ze struktury Songu A a AI návrh scénáře
- `riffs.py` — detekce opakujících se smyček v B podle chroma podobnosti, rytmické gate vzory a stutter/retrigger
- `acid.py` — generátor a syntezátor TB-303 acid linky + export do standardního MIDI (`acid.mid`)
- `dsp.py` — biquad filtry, filtrové sweepy, ping-pong echo, sidechain obálka kicku, snare roll, risery, impakty
- `plots.py` — matplotlib grafy struktury/energie skladeb a časové osy scénáře
- `analysis.py` — přesné BPM, lineární regrese pevného beat gridu, Krumhansl tónina, Camelot kolo, `retime`
- `audio_ops.py` — ffmpeg I/O, `rubberband` time-stretch & pitch-shift, equal-power crossfade
- `stems.py` — separace stemů přes `htdemucs` s diskovou cache v `cache/stems`
- `engine.py` & `recipes.py` — obal nad ACE-Step 1.5 a recepty 1–7
- `lora.py` — trénink vlastního Vibe packu (LoRA) přes Side-Step
- `ui_state.py` — paměť posledního nastavení UI v `cache/last_ui.json`
