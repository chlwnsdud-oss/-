# -*- coding: utf-8 -*-
"""구독 이탈 예측 AI (로지스틱 회귀) - 실행: python churn_app.py  (Python 3 기본 기능만 사용)"""
import csv, math, tkinter as tk
from tkinter import ttk, messagebox, filedialog, font as tkfont

NAMES = ["접속일수", "이용시간", "문의횟수", "월 결제금액"]
UNITS = ["일", "시간", "회", "원"]
LIMIT = [(0, 30), (0, 100), (0, 10), (0, 50000)]   # 입력 허용 범위
INTEGER = [True, False, True, False]
DIRN = ["낮을수록 이탈↑", "적을수록 이탈↑", "많을수록 이탈↑", "높을수록 이탈↑"]
EXAMPLES = {"충성 고객": [26, 60, 0, 9900], "위험 고객": [3, 5, 6, 17000],
            "이상한 값": [99, -5, 50, 900000], "모순 조합": [0, 100, 0, 13000]}
BG, CARD, BORDER, TEXT, MUTED = "#eef3f2", "#ffffff", "#dbe3e1", "#1f2937", "#6b7280"
PRIMARY, DARK, RED, GREEN = "#0f766e", "#134e4a", "#c2410c", "#15803d"
W, H = 1100, 720   # 창 크기(고정 규격)


def sig(z):
    return 1 / (1 + math.exp(-max(-60, min(60, z))))


def mulberry(seed):  # 시드 고정 난수 → 항상 같은 가상 데이터
    s = [seed & 0xFFFFFFFF]
    mul = lambda a, b: (a * b) & 0xFFFFFFFF

    def rnd():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = mul(s[0] ^ (s[0] >> 15), 1 | s[0])
        t = ((t + mul(t ^ (t >> 7), 61 | t)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def make_data(n=600, seed=42):
    r, out = mulberry(seed), []
    for _ in range(n):
        days = math.floor(r() * 30 + .5)
        hours = max(0, min(100, math.floor(days * 2.2 + (r() - .5) * 40 + .5)))
        inq = math.floor(r() * r() * 8)
        pay = math.floor((5000 + r() * 25000) / 100 + .5) * 100
        z = -.18 * (days - 14) - .04 * (hours - 35) + .45 * (inq - 2) + .00015 * (pay - 13000) - 1
        out.append(([days, hours, inq, pay], 1 if r() < sig(z) else 0))
    return out


class Model:
    """입력 4개 x Weight + Bias → Sigmoid → 확률. 예측→오차→수정을 반복(경사하강법)해서 학습."""
    def __init__(s, train, lr, epochs):
        n = len(train)
        s.mu = [sum(d[0][j] for d in train) / n for j in range(4)]
        s.sd = [(sum((d[0][j] - s.mu[j]) ** 2 for d in train) / n) ** .5 or 1 for j in range(4)]
        X = [s.z(d[0]) for d in train]
        Y = [d[1] for d in train]
        s.w, s.b = [0.0] * 4, 0.0
        for _ in range(epochs):
            gw, gb = [0.0] * 4, 0.0
            for x, y in zip(X, Y):
                e = sig(s.b + sum(x[j] * s.w[j] for j in range(4))) - y
                for j in range(4):
                    gw[j] += e * x[j]
                gb += e
            s.w = [w - lr * g / n for w, g in zip(s.w, gw)]
            s.b -= lr * gb / n

    def z(s, x): return [(x[j] - s.mu[j]) / s.sd[j] for j in range(4)]
    def contrib(s, x): return [a * b for a, b in zip(s.z(x), s.w)]
    def prob(s, x): return sig(s.b + sum(s.contrib(x)))


def evaluate(m, data, t):
    TP = FP = TN = FN = 0
    for x, y in data:
        p = m.prob(x) >= t
        if p and y: TP += 1
        elif p: FP += 1
        elif y: FN += 1
        else: TN += 1
    r = lambda a, b: a / b if b else 0
    pre, rec = r(TP, TP + FP), r(TP, TP + FN)
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, acc=r(TP + TN, len(data)), pre=pre, rec=rec,
                f1=r(2 * pre * rec, pre + rec))


def check(raw, rng):
    """입력 검사: 오류(예측 거부)와 경고(참고용)를 나눠서 돌려준다."""
    errs, warns, x = [], [], []
    for j, s in enumerate(raw):
        s = s.strip()
        if not s:
            errs.append(f"{NAMES[j]}: 값을 입력해 주세요"); continue
        try:
            v = float(s)
        except ValueError:
            errs.append(f"{NAMES[j]}: 숫자만 입력할 수 있어요"); continue
        lo, hi = LIMIT[j]
        if not lo <= v <= hi:
            errs.append(f"{NAMES[j]}: {lo}~{hi}{UNITS[j]} 안에서 입력해 주세요")
        elif INTEGER[j] and v != int(v):
            errs.append(f"{NAMES[j]}: 정수만 입력할 수 있어요")
        else:
            x.append(v)
    if not errs:
        if x[1] > x[0] * 24:
            errs.append(f"접속일수와 이용시간이 맞지 않아요: 접속 {x[0]:g}일이면 이용시간은 최대 {x[0] * 24:g}시간이에요")
        for j in range(4):
            if not rng[j][0] <= x[j] <= rng[j][1]:
                warns.append(f"⚠ {NAMES[j]} {x[j]:g}{UNITS[j]}: 학습한 범위({rng[j][0]:g}~{rng[j][1]:g}{UNITS[j]}) 밖이라 참고용이에요")
    return errs, warns, x


class App:
    def __init__(s, root):
        s.root = root
        fams = set(tkfont.families(root))
        s.F = next((f for f in ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR") if f in fams), "TkDefaultFont")
        root.title("구독 이탈 예측 AI"); root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 3)}"); root.minsize(1000, 680)
        s.style()
        hd = tk.Frame(root, bg=PRIMARY, height=68); hd.pack(fill="x"); hd.pack_propagate(False)
        tk.Label(hd, text="구독 이탈 예측 AI", bg=PRIMARY, fg="white", font=s.fnt(20, True)).pack(side="left", padx=(24, 14))
        tk.Label(hd, text="로지스틱 회귀로 다음 달 해지 확률을 계산합니다", bg=PRIMARY, fg="#ccfbf1", font=s.fnt(11)).pack(side="left")
        data = make_data(); s.train, s.test = data[:480], data[480:]
        s.rng = [(min(d[0][j] for d in s.train), max(d[0][j] for d in s.train)) for j in range(4)]
        s.exps, s.cache, s.th = [], {}, tk.DoubleVar(value=0.5)
        nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=16, pady=(12, 16))
        s.tabs = [tk.Frame(nb, bg=BG, padx=14, pady=14) for _ in range(4)]
        for f, n in zip(s.tabs, ("① 예측", "② 성능 평가", "③ 오분류 분석", "④ 실험 기록")):
            nb.add(f, text=n)
        s.build_pred(); s.build_eval(); s.build_err(); s.build_exp()
        s.run_exp(0.1, 600, 0.5, "기본 설정 (중간평가 모델)")
        s.run_exp(0.1, 600, 0.4, "기준값 0.5→0.4: 놓친 이탈 고객(FN)을 줄이기 위해 (4차 과제 분석)")
        s.apply(1)

    # ── 꾸미기 도구 ──
    def fnt(s, size=10, bold=False): return (s.F, size, "bold" if bold else "normal")

    def style(s):
        st = ttk.Style(); st.theme_use("clam")
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", font=s.fnt(11, True), padding=(24, 10), background="#d5e3e0", foreground=MUTED, borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", PRIMARY)])
        st.configure("Treeview", font=s.fnt(10), rowheight=28, background=CARD, fieldbackground=CARD, foreground=TEXT, borderwidth=0)
        st.configure("Treeview.Heading", font=s.fnt(10, True), background="#e3eeec", foreground=DARK, relief="flat", padding=6)
        st.map("Treeview", background=[("selected", PRIMARY)], foreground=[("selected", "white")])

    def card(s, parent, title):
        o = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(o, text=title, bg=CARD, fg=DARK, font=s.fnt(13, True)).pack(anchor="w", padx=20, pady=(16, 6))
        i = tk.Frame(o, bg=CARD); i.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        return o, i

    def button(s, parent, text, cmd, primary=True):
        bg, fg = (PRIMARY, "white") if primary else ("#e3eeec", DARK)
        b = tk.Label(parent, text=text, bg=bg, fg=fg, font=s.fnt(10, True), padx=16, pady=8, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.config(bg=DARK if primary else "#cfe0dc"))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def entry(s, parent, var, width=14):
        return tk.Entry(parent, textvariable=var, width=width, font=s.fnt(11), relief="flat", bg="#f8fafc", fg=TEXT,
                        highlightthickness=1, highlightbackground=BORDER, highlightcolor=PRIMARY, insertbackground=TEXT)

    def table(s, parent, cols, widths):
        wrap = tk.Frame(parent, bg=CARD); wrap.pack(fill="both", expand=True)
        t = ttk.Treeview(wrap, columns=cols, show="headings")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=t.yview); t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); t.pack(side="left", fill="both", expand=True)
        for c, w in zip(cols, widths): t.heading(c, text=c); t.column(c, width=w, anchor="center")
        return t

    # ── ① 예측 ──
    def build_pred(s):
        f = s.tabs[0]; f.columnconfigure((0, 1), weight=1, uniform="c"); f.rowconfigure(0, weight=1)
        lo, L = s.card(f, "고객 정보 입력"); lo.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ro, R = s.card(f, "AI 판단"); ro.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        L.columnconfigure(1, weight=1); s.vars = []
        for j in range(4):
            v = tk.StringVar(value=["12", "30", "2", "13000"][j]); v.trace_add("write", s.refresh); s.vars.append(v)
            tk.Label(L, text=f"{NAMES[j]} ({UNITS[j]})", bg=CARD, fg=TEXT, font=s.fnt(11, True)).grid(row=2 * j, column=0, sticky="w", pady=(10, 0))
            s.entry(L, v).grid(row=2 * j, column=1, sticky="e", pady=(10, 0), ipady=4)
            tk.Label(L, text=f"허용 {LIMIT[j][0]}~{LIMIT[j][1]} · 예상: {DIRN[j]}", bg=CARD, fg=MUTED, font=s.fnt(9)).grid(row=2 * j + 1, column=0, columnspan=2, sticky="w")
        tk.Label(L, text="예시 불러오기", bg=CARD, fg=MUTED, font=s.fnt(10, True)).grid(row=8, column=0, columnspan=2, sticky="w", pady=(22, 4))
        bf = tk.Frame(L, bg=CARD); bf.grid(row=9, column=0, columnspan=2, sticky="ew"); bf.columnconfigure((0, 1), weight=1, uniform="b")
        for i, (n, val) in enumerate(EXAMPLES.items()):
            s.button(bf, n, lambda val=val: s.fill(val), primary=False).grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3)
        tk.Label(R, text="결과를 쓰는 사람: CRM 담당자·마케팅팀 (최종 결정은 사람이 합니다)", bg=CARD, fg=MUTED, font=s.fnt(9)).pack(anchor="w")
        s.res = tk.Label(R, text="", bg=CARD, font=s.fnt(30, True)); s.res.pack(anchor="w", pady=(10, 0))
        s.pl = tk.Label(R, text="", bg=CARD, fg=TEXT, font=s.fnt(12)); s.pl.pack(anchor="w")
        s.cv = tk.Canvas(R, width=460, height=24, bg=CARD, highlightthickness=0); s.cv.pack(anchor="w", pady=8)
        s.act = tk.Label(R, text="", bg=CARD, fg=DARK, font=s.fnt(11, True), wraplength=460, justify="left"); s.act.pack(anchor="w")
        s.msg = tk.Label(R, text="", bg=CARD, fg=RED, font=s.fnt(10), wraplength=460, justify="left"); s.msg.pack(anchor="w", pady=6)
        tk.Label(R, text="왜 그렇게 판단했나? (빨강=이탈 쪽, 초록=유지 쪽)", bg=CARD, fg=MUTED, font=s.fnt(10, True)).pack(anchor="w", pady=(10, 0))
        s.cb = tk.Canvas(R, width=460, height=130, bg=CARD, highlightthickness=0); s.cb.pack(anchor="w")

    def fill(s, val):
        for v, a in zip(s.vars, val): v.set(str(a))

    def refresh(s, *_):
        if not hasattr(s, "model"): return
        s.update_pred(); s.update_eval()

    def update_pred(s):
        errs, warns, x = check([v.get() for v in s.vars], s.rng)
        s.cv.delete("all"); s.cb.delete("all")
        if errs:
            s.res.config(text="예측 불가", fg=RED); s.pl.config(text="")
            s.act.config(text=""); s.msg.config(text="\n".join(errs) + "\n\n값을 고친 뒤 다시 입력해 주세요."); return
        p, t = s.model.prob(x), round(s.th.get(), 2)
        churn, vague = p >= t, .35 <= p <= .65
        s.res.config(text="이탈 위험" if churn else "유지 예상", fg=RED if churn else GREEN)
        s.pl.config(text=f"이탈 확률 {p * 100:.1f}%  (기준값 {t:.2f})")
        s.cv.create_rectangle(0, 5, 460, 19, fill="#e5e7eb", width=0)
        s.cv.create_rectangle(0, 5, 460 * p, 19, fill=PRIMARY, width=0)
        s.cv.create_line(460 * t, 0, 460 * t, 24, width=2, fill=TEXT)
        c = s.model.contrib(x); top = c.index(max(c))
        if vague: a = "→ 애매한 구간: 자동 쿠폰 발송 대신 CRM 담당자가 확인하세요"
        elif not churn: a = "→ 별도 조치 없음 (정기 모니터링)"
        else: a = ["→ 마케팅팀: 이용이 줄어드는 중. 재방문 알림과 할인 쿠폰 대상", "→ 마케팅팀: 이용이 줄어드는 중. 재방문 알림과 할인 쿠폰 대상",
                   "→ CRM 담당자: 불만 가능성이 커 보임. 상담 연결 우선", "→ 마케팅팀: 요금 부담 가능성. 할인 쿠폰 발송 대상"][top]
        s.act.config(text=a); s.msg.config(text="\n".join(warns))
        mx = max(max(abs(v) for v in c), .01)
        for j, v in enumerate(c):
            y = 14 + j * 30
            s.cb.create_text(0, y, text=NAMES[j], anchor="w", fill=TEXT, font=s.fnt(10))
            s.cb.create_rectangle(270, y - 8, 270 + v / mx * 120, y + 8, fill=RED if v > 0 else GREEN, width=0)
            s.cb.create_text(458, y, text=f"{v:+.1f}", anchor="e", fill=MUTED, font=s.fnt(10))

    # ── ② 성능 평가 ──
    def build_eval(s):
        f = s.tabs[1]; f.columnconfigure((0, 1), weight=1, uniform="c"); f.rowconfigure(0, weight=1)
        lo, L = s.card(f, "혼동행렬 (테스트 고객 120명)"); lo.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ro, R = s.card(f, "성능 지표"); ro.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        s.thl = tk.Label(L, text="", bg=CARD, fg=TEXT, font=s.fnt(11, True)); s.thl.pack(anchor="w")
        tk.Scale(L, from_=.1, to=.9, resolution=.05, orient="horizontal", variable=s.th, command=s.refresh, showvalue=0, length=440,
                 bg=CARD, troughcolor="#d1d5db", highlightthickness=0, activebackground=PRIMARY, bd=0).pack(anchor="w", pady=6)
        g = tk.Frame(L, bg=CARD); g.pack(anchor="w", pady=(14, 0)); s.cells = {}
        for c, t in enumerate(("", "AI: 유지", "AI: 이탈")):
            tk.Label(g, text=t, bg=CARD, fg=MUTED, font=s.fnt(10, True), width=14 if c else 10).grid(row=0, column=c)
        rows = (("실제: 유지", (("TN", "맞힘", "#dcfce7"), ("FP", "잘못 경고", "#fee2e2"))),
                ("실제: 이탈", (("FN", "놓침", "#fee2e2"), ("TP", "찾음", "#dcfce7"))))
        for r, (rt, keys) in enumerate(rows, 1):
            tk.Label(g, text=rt, bg=CARD, fg=MUTED, font=s.fnt(10, True), width=10).grid(row=r, column=0)
            for c, (k, cap, col) in enumerate(keys, 1):
                cell = tk.Label(g, text="", bg=col, fg=TEXT, font=s.fnt(13, True), width=14, height=3)
                cell.grid(row=r, column=c, padx=3, pady=3); s.cells[k] = (cell, cap)
        tg = tk.Frame(R, bg=CARD); tg.pack(fill="x"); s.tiles = {}
        for i, (k, n, d) in enumerate((("acc", "전체 정답률", "Accuracy"), ("pre", "경고 정확도", "Precision"),
                                       ("rec", "이탈 발견율", "Recall"), ("f1", "균형 점수", "F1")), 0):
            t = tk.Frame(tg, bg="#f0f7f5"); t.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4); tg.columnconfigure(i % 2, weight=1, uniform="m")
            tk.Label(t, text=n, bg="#f0f7f5", fg=MUTED, font=s.fnt(10)).pack(anchor="w", padx=14, pady=(12, 0))
            s.tiles[k] = tk.Label(t, text="", bg="#f0f7f5", fg=PRIMARY, font=s.fnt(24, True)); s.tiles[k].pack(anchor="w", padx=14)
            tk.Label(t, text=d, bg="#f0f7f5", fg=MUTED, font=s.fnt(9)).pack(anchor="w", padx=14, pady=(0, 12))

    def update_eval(s):
        t = round(s.th.get(), 2); m = evaluate(s.model, s.test, t)
        s.thl.config(text=f"기준값(Threshold) {t:.2f}  ·  이탈 확률이 이 값 이상이면 '이탈'")
        for k, (cell, cap) in s.cells.items(): cell.config(text=f"{m[k]}명\n{cap}")
        for k, lab in s.tiles.items(): lab.config(text=f"{m[k] * 100:.1f}%")
        s.update_err(t)

    # ── ③ 오분류 분석 ──
    def build_err(s):
        o, i = s.card(s.tabs[2], "오분류 분석 (AI가 틀린 고객 살펴보기)"); o.pack(fill="both", expand=True)
        s.es = tk.Label(i, text="", bg=CARD, fg=TEXT, font=s.fnt(11), wraplength=980, justify="left"); s.es.pack(anchor="w", pady=(0, 10))
        s.et = s.table(i, ("유형", "접속", "이용", "문의", "결제", "AI 이탈확률"), (160, 120, 120, 120, 140, 160))

    def update_err(s, t):
        s.et.delete(*s.et.get_children()); wrong = []
        for x, y in s.test:
            p = s.model.prob(x)
            if (p >= t) != bool(y): wrong.append((("놓침(FN)" if y else "잘못 경고(FP)"), x, p))
        for k, x, p in sorted(wrong, key=lambda a: a[0]): s.et.insert("", "end", values=(k, *[f"{v:g}" for v in x], f"{p * 100:.0f}%"))
        v = sum(1 for w in wrong if .35 <= w[2] <= .65)
        s.es.config(text=f"오분류 {len(wrong)}건 중 {v}건({v * 100 // max(len(wrong), 1)}%)이 애매한 구간(확률 35~65%)에 있어요. "
                         "확률이 매우 높거나 낮을 때는 거의 틀리지 않아서, 애매한 고객은 사람이 확인하도록 했습니다.")

    # ── ④ 실험 기록 ──
    def build_exp(s):
        f = s.tabs[3]; o, i = s.card(f, "실험 설정"); o.pack(fill="x")
        row = tk.Frame(i, bg=CARD); row.pack(fill="x"); s.ev = {}
        for n, d in (("학습률", "0.1"), ("반복 횟수", "600"), ("기준값", "0.5")):
            tk.Label(row, text=n, bg=CARD, fg=TEXT, font=s.fnt(11, True)).pack(side="left"); s.ev[n] = tk.StringVar(value=d)
            s.entry(row, s.ev[n], width=8).pack(side="left", padx=(8, 22), ipady=3)
        tk.Label(i, text="변경 이유 (왜 바꿨나요?)", bg=CARD, fg=TEXT, font=s.fnt(11, True)).pack(anchor="w", pady=(12, 4))
        s.reason = tk.StringVar(); s.entry(i, s.reason, width=60).pack(fill="x", ipady=4)
        bf = tk.Frame(i, bg=CARD); bf.pack(anchor="w", pady=(12, 0))
        s.button(bf, "학습하고 기록", s.on_run).pack(side="left", padx=(0, 8))
        s.button(bf, "선택한 실험을 앱에 적용", s.sel_apply, False).pack(side="left", padx=(0, 8))
        s.button(bf, "CSV로 저장", s.save_csv, False).pack(side="left")
        s.status = tk.Label(bf, text="", bg=CARD, fg=MUTED, font=s.fnt(10)); s.status.pack(side="left", padx=14)
        o2, i2 = s.card(f, "실험 기록 (기본 실험 대비 변화)"); o2.pack(fill="both", expand=True, pady=(14, 0))
        s.xt = s.table(i2, ("번호", "학습률", "반복", "기준값", "정확도", "정밀도", "재현율", "F1", "놓침FN", "경고FP", "기본 대비 F1", "변경 이유"),
                       (50, 60, 60, 60, 70, 70, 70, 60, 70, 70, 100, 260))
        s.xt.column("변경 이유", anchor="w", stretch=True)

    def on_run(s):
        try:
            lr, ep, th = float(s.ev["학습률"].get()), int(s.ev["반복 횟수"].get()), float(s.ev["기준값"].get())
        except ValueError:
            return messagebox.showerror("입력 오류", "학습률·기준값은 숫자, 반복 횟수는 정수로 입력해 주세요.")
        if not (0.001 <= lr <= 5 and 1 <= ep <= 3000 and 0.05 <= th <= 0.95):
            return messagebox.showerror("입력 오류", "학습률 0.001~5, 반복 횟수 1~3000, 기준값 0.05~0.95 안에서 입력해 주세요.")
        if not s.reason.get().strip():
            return messagebox.showerror("입력 오류", "왜 바꿨는지 변경 이유를 적어 주세요.")
        s.status.config(text="학습 중..."); s.root.update_idletasks()
        s.run_exp(lr, ep, th, s.reason.get().strip()); s.status.config(text=f"#{len(s.exps)} 기록 완료")

    def run_exp(s, lr, ep, th, why):
        if (lr, ep) not in s.cache: s.cache[(lr, ep)] = Model(s.train, lr, ep)
        m = s.cache[(lr, ep)]; r = evaluate(m, s.test, th)
        s.exps.append(dict(lr=lr, ep=ep, th=th, r=r, why=why, model=m))
        base = s.exps[0]["r"]["f1"]
        s.xt.insert("", "end", values=(len(s.exps), lr, ep, th, *[f"{r[k] * 100:.1f}%" for k in ("acc", "pre", "rec", "f1")],
                                       r["FN"], r["FP"], f"{(r['f1'] - base) * 100:+.1f}%p", why))
        try: s.save_csv(auto=True)
        except OSError: pass

    def sel_apply(s):
        sel = s.xt.selection()
        if not sel: return messagebox.showinfo("안내", "표에서 적용할 실험을 먼저 선택해 주세요.")
        s.apply(s.xt.index(sel[0]))

    def apply(s, i):
        e = s.exps[i]; s.model = e["model"]; s.th.set(e["th"]); s.refresh()

    def save_csv(s, auto=False):
        path = "experiments.csv" if auto else filedialog.asksaveasfilename(defaultextension=".csv", initialfile="experiments.csv")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp); w.writerow(["번호", "학습률", "반복", "기준값", "정확도", "정밀도", "재현율", "F1", "FN", "FP", "변경 이유"])
            for i, e in enumerate(s.exps, 1):
                r = e["r"]; w.writerow([i, e["lr"], e["ep"], e["th"], *[round(r[k], 3) for k in ("acc", "pre", "rec", "f1")], r["FN"], r["FP"], e["why"]])
        if not auto: messagebox.showinfo("저장 완료", path)


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
