"""Renders the accuracy dashboard shown on the web page.

Numbers are never hard-coded: everything is read from results/evaluation.json,
which is produced by `python -m src.evaluate`.
"""
import json
import os

RESULTS_PATH = os.environ.get("EVAL_RESULTS", "results/evaluation.json")
QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def load_results():
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _card(title, value, sub=""):
    return (
        '<div style="flex:1 1 170px;padding:14px 16px;border:1px solid var(--border-color-primary);'
        'border-radius:12px;background:var(--background-fill-secondary);">'
        f'<div style="font-size:12px;opacity:.7">{title}</div>'
        f'<div style="font-size:26px;font-weight:700;margin-top:2px">{value}</div>'
        f'<div style="font-size:11px;opacity:.6;margin-top:2px">{sub}</div></div>'
    )


def _no_results():
    return (
        '<div style="padding:14px 16px;border:1px dashed var(--border-color-primary);border-radius:12px;">'
        "📊 <b>Accuracy not computed yet.</b> Run "
        "<code>python -m src.evaluate --emopia_dir &lt;path&gt; ...</code> and refresh."
        "</div>"
    )


def render_banner(res):
    """One-line headline badge for the top of the page."""
    if not res:
        return _no_results()
    h = res.get("headline", {})
    return (
        '<div style="display:flex;align-items:center;gap:12px;padding:10px 16px;'
        'border:1px solid var(--border-color-primary);border-radius:12px;'
        'background:var(--background-fill-secondary);">'
        f'<span style="font-size:28px;font-weight:800">{_pct(h.get("accuracy"))}</span>'
        f'<span style="opacity:.75">overall accuracy ({h.get("definition", "")}) · '
        f'random guessing = {_pct(res.get("chance_level", 0.25))} · see the “📊 Model Accuracy” tab</span></div>'
    )


def render_accuracy_html(res):
    if not res:
        return _no_results()

    h = res.get("headline", {})
    face, text = res.get("face"), res.get("text")
    gen, e2e, clf = res.get("generation", {}), res.get("end_to_end", {}), res.get("classifier", {})

    parts = [
        '<div style="display:flex;flex-direction:column;gap:18px">',
        '<div style="padding:22px;border-radius:16px;border:1px solid var(--border-color-primary);'
        'background:var(--background-fill-secondary);text-align:center">'
        '<div style="font-size:13px;opacity:.7">OVERALL SYSTEM ACCURACY</div>'
        f'<div style="font-size:56px;font-weight:800;line-height:1.1">{_pct(h.get("accuracy"))}</div>'
        f'<div style="font-size:12px;opacity:.65">definition: {h.get("definition", "")} · '
        f'chance level {_pct(res.get("chance_level", 0.25))} · evaluated {res.get("generated_at", "")}</div></div>',
    ]

    def section(title, cards):
        parts.append(f'<div><div style="font-weight:700;margin-bottom:8px">{title}</div>'
                     f'<div style="display:flex;flex-wrap:wrap;gap:10px">{"".join(cards)}</div></div>')

    pc = []
    if face:
        pc += [_card("Face · 7-emotion accuracy", _pct(face.get("emotion_accuracy")), f'n={face.get("n")}'),
               _card("Face · macro-F1", _pct(face.get("emotion_macro_f1")), "class-balanced"),
               _card("Face · quadrant accuracy", _pct(face.get("quadrant_accuracy")), "emotion → Q1–Q4")]
    if text:
        pc += [_card("Text · quadrant accuracy", _pct(text.get("quadrant_accuracy")), f'n={text.get("n")}'),
               _card("Text · macro-F1", _pct(text.get("quadrant_macro_f1")), "class-balanced")]
    if pc:
        section("1 · Perception (does it read the emotion correctly?)", pc)

    section("2 · Music generation (does the music match the requested quadrant?)", [
        _card("Conditioning accuracy", _pct(gen.get("conditioning_accuracy")), f'n={gen.get("n")} generated clips'),
        _card("Valence-axis accuracy", _pct(gen.get("valence_accuracy")), "positive vs negative"),
        _card("Arousal-axis accuracy", _pct(gen.get("arousal_accuracy")), "high vs low energy"),
        _card("Valid MIDI rate", _pct(gen.get("valid_midi_rate")), "playable, non-empty output"),
        _card("Real-music reference", _pct(clf.get("real_music_accuracy")), "judge's accuracy on real EMOPIA"),
    ])

    ec = []
    for k, label in (("face", "Face → music"), ("text", "Text → music")):
        v = e2e.get(k)
        if v:
            ec.append(_card(label, _pct(v.get("accuracy")), f'n={v.get("n")}'))
    if ec:
        section("3 · End-to-end (input → music → judged emotion = true emotion)", ec)

    cm = gen.get("confusion")
    if cm:
        rows = "".join(
            f'<tr><td style="padding:4px 10px;font-weight:600">{QUADRANTS[i]}</td>'
            + "".join(f'<td style="padding:4px 10px;text-align:center">{v}</td>' for v in row) + "</tr>"
            for i, row in enumerate(cm)
        )
        head = "".join(f'<th style="padding:4px 10px">{q}</th>' for q in QUADRANTS)
        pq = gen.get("per_quadrant", {})
        parts.append(
            '<div><div style="font-weight:700;margin-bottom:8px">Confusion matrix (rows = requested, columns = judged)</div>'
            f'<table style="border-collapse:collapse"><tr><th></th>{head}</tr>{rows}</table>'
            '<div style="font-size:12px;opacity:.7;margin-top:6px">Per-quadrant accuracy: '
            + " · ".join(f"{q}: {_pct(pq.get(q))}" for q in QUADRANTS) + "</div></div>"
        )

    parts.append(
        '<div style="font-size:12px;opacity:.65">Method: music is judged by an independent classifier trained on real '
        "EMOPIA clips (tempo, note density, dynamics, pitch, major/minor mode). Failed or empty generations count as wrong.</div></div>"
    )
    return "".join(parts)
