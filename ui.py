"""Small, reusable presentation primitives for the Vermillion dashboard."""

from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components


def load_css(css_path: str = "styles.css") -> None:
    """Load the local stylesheet once for the current Streamlit page."""
    path = Path(__file__).parent / css_path
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def header() -> None:
    """Render a quiet site header."""
    st.markdown("""
    <header class="site-header">
      <div class="brand">Vermillion</div>
      <div class="header-context">Retail analytics</div>
    </header>
    """, unsafe_allow_html=True)


def hero() -> None:
    """Render the focused dashboard introduction."""
    st.markdown("""
    <section class="hero">
      <div>
        <h1>Retail intelligence,<br>without the clutter.</h1>
      </div>
      <p class="hero-copy">Upload a transaction file to review revenue, customer behaviour, and product performance in one place.</p>
    </section>
    """, unsafe_allow_html=True)


def render_3d_sphere(items: list[dict[str, str]], height: int = 390) -> None:
    """Render a lightweight, draggable orbit of the highest-revenue products."""
    payload = json.dumps(items)
    components.html(f"""
    <style>
      html,body {{ margin:0; background:#090a0b; overflow:hidden; }}
      #scene {{ height:{height}px; position:relative; perspective:850px; cursor:grab; user-select:none; }}
      #scene:active {{ cursor:grabbing; }}
      #orbit {{ position:absolute; left:50%; top:48%; transform-style:preserve-3d; }}
      .node {{ position:absolute; width:132px; transform-origin:center; color:#f1f3ef; font:12px Inter,Arial,sans-serif; pointer-events:none; }}
      .node span {{ display:block; color:#8c928d; font-size:10px; margin-top:4px; }}
      .hint {{ position:absolute; left:0; bottom:12px; color:#8c928d; font:11px Inter,Arial,sans-serif; }}
    </style>
    <div id="scene"><div id="orbit"></div><div class="hint">Drag to explore product performance</div></div>
    <script>
      const items = {payload}; const orbit = document.getElementById('orbit'); const scene = document.getElementById('scene');
      const radius = Math.min(155, window.innerWidth * .27); const points = []; const golden = Math.PI * (3 - Math.sqrt(5));
      items.forEach((item, i) => {{
        const y = 1 - (i / Math.max(items.length - 1, 1)) * 2; const spread = Math.sqrt(1 - y * y); const theta = golden * i;
        const node = document.createElement('div'); node.className = 'node';
        const title = document.createElement('div'); title.textContent = item.title || 'Product';
        const value = document.createElement('span'); value.textContent = item.value || ''; node.append(title, value); orbit.append(node);
        points.push({{node, x:Math.cos(theta)*spread*radius, y:y*radius, z:Math.sin(theta)*spread*radius}});
      }});
      let targetX = 8, targetY = 0, x = 8, y = 0, down = false, lastX = 0, lastY = 0;
      function frame() {{
        if (!down) targetY += .12; x += (targetX-x)*.08; y += (targetY-y)*.08;
        const rx=x*Math.PI/180, ry=y*Math.PI/180, cx=Math.cos(rx), sx=Math.sin(rx), cy=Math.cos(ry), sy=Math.sin(ry);
        points.forEach(p => {{ const px=p.x*cy+p.z*sy, pz=-p.x*sy+p.z*cy, py=p.y*cx-pz*sx, z=p.y*sx+pz*cx; const fade=Math.max(.24, Math.min(1,(z+radius)/(radius*1.6))); p.node.style.transform=`translate3d(${{px-66}}px,${{py}}px,${{z}}px)`; p.node.style.opacity=fade; }});
        requestAnimationFrame(frame);
      }} frame();
      scene.addEventListener('pointerdown', e => {{ down=true; lastX=e.clientX; lastY=e.clientY; scene.setPointerCapture(e.pointerId); }});
      scene.addEventListener('pointermove', e => {{ if(!down) return; targetY+=(e.clientX-lastX)*.45; targetX=Math.max(-50,Math.min(50,targetX-(e.clientY-lastY)*.45)); lastX=e.clientX; lastY=e.clientY; }});
      scene.addEventListener('pointerup', () => down=false);
    </script>
    """, height=height, scrolling=False)
