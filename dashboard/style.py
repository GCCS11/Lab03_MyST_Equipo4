"""Look of the app: navy background with soft glows, glass panels, pill controls and staggered
entrances. Only CSS and small HTML snippets; the charts and numbers come from figures.py.

Motion only animates transform, opacity and filter, and is skipped with prefers-reduced-motion.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
:root{--glass:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
  --line:rgba(255,255,255,.09);--ease:cubic-bezier(.16,1,.3,1);--blue:#4f7dff;--green:#2fd6a0}
.stApp{font-family:Outfit,system-ui,-apple-system,"Segoe UI",sans-serif;
  background:radial-gradient(1100px 620px at 12% 6%,rgba(46,96,255,.30),transparent 62%),
             radial-gradient(900px 640px at 92% 28%,rgba(0,190,170,.13),transparent 60%),
             radial-gradient(1200px 760px at 50% 108%,rgba(34,70,220,.30),transparent 62%),#050b1e}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.2rem;max-width:1440px}
h1{font-weight:600!important;letter-spacing:-.02em}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(12,26,61,.92),rgba(6,13,34,.92));
  border-right:1px solid var(--line)}
[data-testid="stPlotlyChart"],[data-testid="stExpander"] details,[data-testid="stMetric"],
[data-testid="stDataFrame"]{background:var(--glass);border:1px solid var(--line);border-radius:18px;
  box-shadow:0 14px 44px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.05)}
[data-testid="stPlotlyChart"]{padding:12px 14px}
/* the frame makes the chart a few px taller than its slot; without this the slot becomes a tiny
   scroll box that swallows the mouse wheel and the page stops scrolling over a chart */
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]){overflow:visible!important}
[data-testid="stMetricValue"]{font-size:1.9rem;font-variant-numeric:tabular-nums}
[data-testid="stMetricValue"] div{overflow:visible!important;text-overflow:clip!important}
.stButton button,[data-testid="stBaseButton-secondary"]{border-radius:12px;background:var(--glass);
  border:1px solid var(--line);transition:transform .18s var(--ease),border-color .18s,box-shadow .18s}
[data-testid="stButtonGroup"] button{border-radius:999px!important;transition:background .22s var(--ease),
  box-shadow .22s var(--ease),transform .16s var(--ease)}
[data-testid="stButtonGroup"] button[kind*="Active"]{background:linear-gradient(180deg,#5b86ff,#3a5cf0)!important;
  color:#fff!important;box-shadow:0 0 22px rgba(79,125,255,.55)}
@media (hover:hover) and (pointer:fine){
  .stButton button:hover,[data-testid="stButtonGroup"] button:hover{transform:translateY(-1px);
    border-color:rgba(120,160,255,.55);box-shadow:0 0 18px rgba(79,125,255,.35)}}
.topbar{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:16px;
  background:var(--glass);border:1px solid var(--line);margin-bottom:.6rem}
.crumbs{font-size:.92rem;color:rgba(232,237,248,.62)} .crumbs b{color:#e8edf8;font-weight:500}
.crumbs i{font-style:normal;margin:0 8px;opacity:.5}
.spacer{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;font-size:.8rem;
  border:1px solid var(--line);background:rgba(255,255,255,.04)}
.pill.live{color:#3ee0a8;background:rgba(47,214,160,.12);border-color:rgba(62,224,168,.32)}
.pill.live::before{content:"";width:7px;height:7px;border-radius:50%;background:#3ee0a8;box-shadow:0 0 8px #3ee0a8}
.pill.blue{color:#fff;background:linear-gradient(180deg,#5b86ff,#3a5cf0);border-color:transparent;
  box-shadow:0 0 16px rgba(79,125,255,.45)}
.hero h1{font-size:2.6rem;margin:.2rem 0 0}
.hero p{color:rgba(232,237,248,.62);margin:.1rem 0 .4rem}
@media (prefers-reduced-motion:no-preference){
  .topbar{animation:rise .9s var(--ease) both}
  [data-testid="stPlotlyChart"]{animation:lift 1.1s var(--ease) .18s both}
  @keyframes rise{from{opacity:0;transform:translateY(12px);filter:blur(4px)}to{opacity:1;transform:none;filter:none}}
  @keyframes lift{from{opacity:0;transform:translateY(18px) scale(.985)}to{opacity:1;transform:none}}}
</style>
"""


def topbar(crumbs, pills):
    """Breadcrumbs on the left and status pills on the right, like an app header."""
    trail = "<i>›</i>".join(f"<b>{c}</b>" if i == len(crumbs) - 1 else c for i, c in enumerate(crumbs))
    badges = "".join(f"<span class='pill {kind}'>{text}</span>" for text, kind in pills)
    return f"<div class='topbar'><span class='crumbs'>{trail}</span><span class='spacer'></span>{badges}</div>"
