"""Draw the README figures as SVG, one light and one dark file each, from results/summary/*.csv.
  python figures/make_figures.py
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = 760
FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
THEMES = {  # page = GitHub's background, used for the 2px rings around markers
    'light': dict(page='#ffffff', ink='#0b0b0b', ink2='#52514e', muted='#8f8d86', grid='#e1e0d9', axis='#c3c2b7',
                  accent='#2a78d6', accent2='#9cc1ee'),
    'dark': dict(page='#0d1117', ink='#f0efec', ink2='#c3c2b7', muted='#6f6d68', grid='#2c2c2a', axis='#4a4a46',
                 accent='#3987e5', accent2='#28507f'),
}

# Series shown, in drawing order (the accent last so it sits on top): run name in results/, label, style.
ONE_USER = [
    ('sglang-draft-off', 'SGLang', 'muted'),
    ('llamacpp', 'llama.cpp', 'muted'),
    ('vllm-draft-off', 'vLLM, draft off', 'muted'),
    ('strata-1slot-draft-off', 'Strata, draft off', 'muted'),
    ('vllm-draft-on', 'vLLM, draft on', 'muted'),
    ('strata-1slot-draft-on', 'Strata, draft on', 'accent'),
]
MANY_USERS = [  # several runs feed one series (vLLM's saturation points were separate runs)
    (('sglang-draft-off', 'sglang-draft-off-users-4-32'), 'SGLang', 'muted', '6 4'),
    (('llamacpp',), 'llama.cpp', 'muted', '2 3'),
    (('strata-4slots',), 'Strata, 4 slots', 'muted', None),
    (('vllm-draft-on',), 'vLLM, draft on', 'accent2', None),
    (('vllm-draft-off', 'vllm-draft-off-saturation-64-128', 'vllm-draft-off-saturation-256'), 'vLLM, draft off', 'accent', None),
]


def rows(name):
    with open(os.path.join(ROOT, 'results', 'summary', name)) as fh:
        return list(csv.DictReader(fh))


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Svg:
    def __init__(self, h, t):
        self.h, self.t, self.parts = h, t, []

    def text(self, x, y, s, size=13, fill='ink', anchor='start', weight=400):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.t[fill]}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke='grid', width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{self.t[stroke]}" stroke-width="{width}"{d}/>')

    def polyline(self, pts, stroke, width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{self.t[stroke]}" stroke-width="{width}" '
                          f'stroke-linejoin="round"{d}/>')

    def dot(self, x, y, fill, r=4):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.t[fill]}" '
                          f'stroke="{self.t["page"]}" stroke-width="2"/>')

    def title(self, title, sub):
        self.text(0, 20, title, size=16, weight=600)
        self.text(0, 41, sub, size=13, fill='ink2')

    def save(self, name, alt):
        body = '\n'.join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" viewBox="0 0 {W} {self.h}" '
               f'font-family="{FONT}" role="img" aria-label="{esc(alt)}">\n<title>{esc(alt)}</title>\n{body}\n</svg>\n')
        with open(os.path.join(HERE, name), 'w', encoding='utf-8') as fh:
            fh.write(svg)


def one_user(theme):
    t = THEMES[theme]
    data = rows('one_user.csv')
    lens = ['1k', '8k', '32k']
    val = {(r['run'], r['prompt']): float(r['decode_tps']) for r in data}
    s = Svg(400, t)
    title = 'One user: Strata with its draft layer decodes fastest; llama.cpp slows as prompts grow'
    s.title(title, 'Decode speed per user in tokens/s by prompt length; 512-token answers, greedy, thinking off')
    left, right, y0, h = 46, 500, 76, 270
    x = lambda i: left + i * (right - left) / (len(lens) - 1)
    y = lambda v: y0 + h - v / 200 * h
    for v in (0, 50, 100, 150, 200):
        s.line(left, y(v), right, y(v))
        s.text(left - 10, y(v) + 4, v, size=12, fill='ink2', anchor='end')
    for i, l in enumerate(lens):
        s.text(x(i), y0 + h + 24, f'{l.upper()} prompt', size=12, fill='ink2', anchor='middle')
    for run, label, style in ONE_USER:
        pts = [(x(i), y(val[(run, l)])) for i, l in enumerate(lens)]
        accent = style == 'accent'
        s.polyline(pts, style, width=3 if accent else 2)
        for px, py in pts:
            s.dot(px, py, style, r=5 if accent else 4)
        last = val[(run, lens[-1])]
        s.text(right + 14, pts[-1][1] + 4, f'{label}: {last:.0f}', size=13, fill='ink' if accent else 'ink2',
               weight=600 if accent else 400)
    s.save(f'one_user_{theme}.svg', title)


def many_users(theme):
    t = THEMES[theme]
    data = rows('many_users.csv')
    s = Svg(440, t)
    title = 'Many users: vLLM scales to about 1,400 tokens/s; the others level off below 320'
    s.title(title, 'Total output tokens/s by concurrent users; each user sends a 1K prompt and gets a 512-token answer')
    left, right, y0, h, ymax = 56, 560, 76, 300, 1500
    users = sorted({int(r['users']) for r in data})
    x = lambda u: left + math.log2(u) / math.log2(users[-1]) * (right - left)
    y = lambda v: y0 + h - v / ymax * h
    for v in (0, 500, 1000, 1500):
        s.line(left, y(v), right, y(v))
        s.text(left - 10, y(v) + 4, f'{v:,}', size=12, fill='ink2', anchor='end')
    for u in users:
        s.text(x(u), y0 + h + 22, u, size=12, fill='ink2', anchor='middle')
    s.text((left + right) / 2, y0 + h + 44, 'concurrent users (each step doubles)', size=12, fill='ink2', anchor='middle')
    series = {}
    for runs, label, style, dash in MANY_USERS:
        pts = sorted((int(r['users']), float(r['total_tps'])) for r in data if r['run'] in runs)
        series[label] = pts
    vllm = series['vLLM, draft off']
    plateau = [v for u, v in vllm if u >= 64]
    level = sum(plateau) / len(plateau)
    s.line(x(64), y(level), right, y(level), stroke='axis', dash='4 4')
    s.text(right + 8, y(level) + 18, f'plateau {level:,.0f}', size=12, fill='ink2')
    stack_y = {'SGLang': 296, 'llama.cpp': 318, 'Strata, 4 slots': 340}
    for runs, label, style, dash in MANY_USERS:
        pts = [(x(u), y(v)) for u, v in series[label]]
        accent = style == 'accent'
        s.polyline(pts, style, width=3 if accent else 2, dash=dash)
        for px, py in pts:
            s.dot(px, py, style, r=4)
        u, v = series[label][-1]
        if label in stack_y:   # the three slow engines sit close together: labels in open space with a leader line
            ly = stack_y[label]
            s.line(x(u) + 6, y(v), right + 10, ly - 4, stroke='grid')
            s.text(right + 14, ly, f'{label}: {v:,.0f} at {u}', size=13, fill='ink2')
        elif accent:
            s.text(x(u), y(v) - 14, f'{label}: {v:,.0f} at {u}', size=13, fill='ink', anchor='end', weight=600)
        else:
            s.text(x(u) + 10, y(v) + 24, f'{label}: {v:,.0f} at {u}', size=13, fill='ink2')
    s.save(f'many_users_{theme}.svg', title)


if __name__ == '__main__':
    for theme in THEMES:
        one_user(theme)
        many_users(theme)
    print('figures written')
