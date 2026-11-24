# -*- coding: utf-8 -*-
"""구독 이탈 예측 AI (로지스틱 회귀) - 실행: python churn_app.py  (Python 3 기본 기능만 사용)"""
import csv, math, tkinter as tk
from tkinter import ttk, messagebox, filedialog

NAMES = ["접속일수", "이용시간", "문의횟수", "월 결제금액"]
UNITS = ["일", "시간", "회", "원"]
LIMIT = [(0, 30), (0, 100), (0, 10), (0, 50000)]   # 입력 허용 범위
INTEGER = [True, False, True, False]
DIRN = ["낮을수록 이탈↑", "적을수록 이탈↑", "많을수록 이탈↑", "높을수록 이탈↑"]
EXAMPLES = {"충성 고객": [26, 60, 0, 9900], "위험 고객": [3, 5, 6, 17000],
            "이상한 값": [99, -5, 50, 900000], "모순 조합": [0, 100, 0, 13000]}
RED, GREEN, BLUE, GRAY = "#c2410c", "#15803d", "#0f766e", "#6b7280"


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
        root.title("구독 이탈 예측 AI"); root.geometry("1040x720")
        data = make_data(); s.train, s.test = data[:480], data[480:]
        s.rng = [(min(d[0][j] for d in s.train), max(d[0][j] for d in s.train)) for j in range(4)]
        s.exps, s.cache, s.th = [], {}, tk.DoubleVar(value=0.5)
        nb = ttk.Notebook(root); nb.pack(fill="both", expand=True)
        s.tabs = [ttk.Frame(nb, padding=14) for _ in range(4)]
        for f, n in zip(s.tabs, ("① 예측", "② 성능 평가", "③ 오분류 분석", "④ 실험 기록")):
            nb.add(f, text=n)
        s.build_pred(); s.build_eval(); s.build_err(); s.build_exp()
        s.run_exp(0.1, 600, 0.5, "기본 설정 (중간평가 모델)")
        s.run_exp(0.1, 600, 0.4, "기준값 0.5→0.4: 놓친 이탈 고객(FN)을 줄이기 위해 (4차 과제 분석)")
        s.apply(1)

    # ── ① 예측 ──
    def build_pred(s):
        f = s.tabs[0]; f.columnconfigure(1, weight=1)
        L = ttk.LabelFrame(f, text="고객 정보 입력", padding=12); L.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        s.vars = []
        for j in range(4):
            v = tk.StringVar(value=["12", "30", "2", "13000"][j]); v.trace_add("write", s.refresh); s.vars.append(v)
            ttk.Label(L, text=f"{NAMES[j]} ({UNITS[j]})").grid(row=2 * j, column=0, sticky="w")
            ttk.Entry(L, textvariable=v, width=12).grid(row=2 * j, column=1, padx=6, pady=(6, 0))
            ttk.Label(L, text=f"{LIMIT[j][0]}~{LIMIT[j][1]} · 예상: {DIRN[j]}", foreground=GRAY).grid(row=2 * j + 1, column=0, columnspan=2, sticky="w")
        for i, (n, val) in enumerate(EXAMPLES.items()):
            ttk.Button(L, text=n, command=lambda val=val: [v.set(str(a)) for v, a in zip(s.vars, val)]).grid(row=8 + i // 2, column=i % 2, pady=4, sticky="ew")
        R = ttk.LabelFrame(f, text="AI 판단", padding=12); R.grid(row=0, column=1, sticky="nsew")
        ttk.Label(R, text="결과를 쓰는 사람: CRM 담당자·마케팅팀 (최종 결정은 사람이 합니다)", foreground=GRAY).pack(anchor="w")
        s.res = tk.Label(R, text="", font=("", 28, "bold")); s.res.pack(anchor="w", pady=(8, 0))
        s.pl = ttk.Label(R, text=""); s.pl.pack(anchor="w")
        s.cv = tk.Canvas(R, width=420, height=24, highlightthickness=0); s.cv.pack(anchor="w", pady=6)
        s.act = ttk.Label(R, text="", wraplength=440); s.act.pack(anchor="w")
        s.msg = tk.Label(R, text="", fg=RED, justify="left", wraplength=440); s.msg.pack(anchor="w", pady=6)
        ttk.Label(R, text="왜 그렇게 판단했나? (빨강=이탈 쪽, 초록=유지 쪽)").pack(anchor="w", pady=(10, 0))
        s.cb = tk.Canvas(R, width=440, height=130, highlightthickness=0); s.cb.pack(anchor="w")

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
        s.cv.create_rectangle(0, 5, 420, 19, fill="#e5e7eb", width=0)
        s.cv.create_rectangle(0, 5, 420 * p, 19, fill=BLUE, width=0)
        s.cv.create_line(420 * t, 0, 420 * t, 24, width=2)
        c = s.model.contrib(x); top = c.index(max(c))
        if vague: a = "→ 애매한 구간: 자동 쿠폰 발송 대신 CRM 담당자가 확인하세요"
        elif not churn: a = "→ 별도 조치 없음 (정기 모니터링)"
        else: a = ["→ 마케팅팀: 이용이 줄어드는 중. 재방문 알림과 할인 쿠폰 대상", "→ 마케팅팀: 이용이 줄어드는 중. 재방문 알림과 할인 쿠폰 대상",
                   "→ CRM 담당자: 불만 가능성이 커 보임. 상담 연결 우선", "→ 마케팅팀: 요금 부담 가능성. 할인 쿠폰 발송 대상"][top]
        s.act.config(text=a); s.msg.config(text="\n".join(warns))
        mx = max(max(abs(v) for v in c), .01)
        for j, v in enumerate(c):
            y = 10 + j * 30
            s.cb.create_text(0, y, text=NAMES[j], anchor="w")
            s.cb.create_rectangle(215, y - 7, 215 + v / mx * 110, y + 7, fill=RED if v > 0 else GREEN, width=0)
            s.cb.create_text(435, y, text=f"{v:+.1f}", anchor="e")

    # ── ② 성능 평가 ──
    def build_eval(s):
        f = s.tabs[1]
        ttk.Label(f, text="테스트 고객 120명 (학습에 쓰지 않은 데이터)").pack(anchor="w")
        s.thl = ttk.Label(f, text="기준값", font=("", 11, "bold")); s.thl.pack(anchor="w", pady=(8, 0))
        ttk.Scale(f, from_=.1, to=.9, variable=s.th, command=s.refresh, length=400).pack(anchor="w")
        s.cm = tk.Label(f, justify="left", font=("", 12)); s.cm.pack(anchor="w", pady=14)
        s.mt = tk.Label(f, justify="left", font=("", 12)); s.mt.pack(anchor="w")

    def update_eval(s):
        t = round(s.th.get(), 2); m = evaluate(s.model, s.test, t)
        s.thl.config(text=f"기준값(Threshold): {t:.2f}  (이탈 확률이 이 값 이상이면 '이탈')")
        s.cm.config(text=f"                 AI: 유지            AI: 이탈\n실제: 유지     ✅ {m['TN']}명 맞힘      ❌ {m['FP']}명 잘못 경고\n"
                         f"실제: 이탈     ❌ {m['FN']}명 놓침      ✅ {m['TP']}명 찾음")
        s.mt.config(text=f"전체 정답률(Accuracy)   {m['acc'] * 100:.1f}%\n경고 정확도(Precision)  {m['pre'] * 100:.1f}%\n"
                         f"이탈 발견율(Recall)    {m['rec'] * 100:.1f}%\n균형 점수(F1)          {m['f1'] * 100:.1f}%")
        s.update_err(t)

    # ── ③ 오분류 분석 ──
    def build_err(s):
        f = s.tabs[2]
        s.es = ttk.Label(f, text="", wraplength=950); s.es.pack(anchor="w")
        cols = ("유형", "접속", "이용", "문의", "결제", "AI 이탈확률")
        s.et = ttk.Treeview(f, columns=cols, show="headings", height=18)
        for c in cols: s.et.heading(c, text=c); s.et.column(c, width=120, anchor="center")
        s.et.pack(fill="both", expand=True, pady=8)

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
        f = s.tabs[3]; box = ttk.Frame(f); box.pack(fill="x")
        s.ev = {}
        for i, (n, d) in enumerate((("학습률", "0.1"), ("반복 횟수", "600"), ("기준값", "0.5"))):
            ttk.Label(box, text=n).grid(row=0, column=2 * i, padx=(0, 4)); s.ev[n] = tk.StringVar(value=d)
            ttk.Entry(box, textvariable=s.ev[n], width=8).grid(row=0, column=2 * i + 1, padx=(0, 12))
        ttk.Label(f, text="변경 이유 (왜 바꿨나요?)").pack(anchor="w", pady=(10, 0))
        s.reason = tk.StringVar(); ttk.Entry(f, textvariable=s.reason).pack(fill="x")
        b = ttk.Frame(f); b.pack(anchor="w", pady=8)
        ttk.Button(b, text="학습하고 기록", command=s.on_run).pack(side="left", padx=(0, 6))
        ttk.Button(b, text="선택한 실험을 앱에 적용", command=lambda: s.sel_apply()).pack(side="left", padx=(0, 6))
        ttk.Button(b, text="CSV로 저장", command=s.save_csv).pack(side="left")
        s.status = ttk.Label(b, text="", foreground=GRAY); s.status.pack(side="left", padx=10)
        cols = ("번호", "학습률", "반복", "기준값", "정확도", "정밀도", "재현율", "F1", "놓침FN", "경고FP", "기본 대비 F1", "변경 이유")
        s.xt = ttk.Treeview(f, columns=cols, show="headings", height=14)
        for c in cols: s.xt.heading(c, text=c); s.xt.column(c, width=62, anchor="center")
        s.xt.column("변경 이유", width=330, anchor="w"); s.xt.column("기본 대비 F1", width=90)
        s.xt.pack(fill="both", expand=True)

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
