"""水管过河（引导模式）
固定使用文章的两轮西岸连接方式，只要求输入东岸注水/出水记录
每行一条：8: 3 / 8 3 / 8 -> 3 / 9: -；空行结束一轮
输入有误会提示并重新输入本轮。
"""
import re
import sys
from collections import Counter

def ints(s):
    """从任意字符串里提取所有整数"""
    return [int(x) for x in re.findall(r"\d+", s)]

def article_groups(n):
    """原文章的两轮固定连接方式；未列出的端口由 build 补成单点组。"""
    if n < 3:
        raise ValueError("引导模式至少需要 3 根水管")
    if n % 2 == 0:
        return (
            [[a, a + 1] for a in range(1, n - 1, 2)],
            [[a, a + 1] for a in range(2, n - 1, 2)],
        )
    return (
        [[a, a + 1] for a in range(1, n, 2)],
        [[a, a + 1] for a in range(2, n, 2)],
    )


def describe(pairs, n):
    """把一轮连接方式写成一行提示。"""
    used = {x for g in pairs for x in g}
    text = "，".join("-".join(map(str, g)) for g in pairs)
    rest = [x for x in range(1, n + 1) if x not in used]
    if rest:
        text += "；不连接：" + "、".join(f"西{x}" for x in rest)
    return text


def tests(n):
    """先整轮读完再解析：某行出错时本轮输入已消费，方便整轮重输。"""
    lines = []
    while (s := input().strip()):
        lines.append(s)
    if not lines:
        raise ValueError("本轮没有任何观测，请至少输入一条")

    ts = []
    for no, s in enumerate(lines, 1):
        try:
            m = re.match(r"[东西]?\s*(\d+)\s*(?::|：|->|→)\s*(.*)", s)
            if m:
                e, rest = int(m[1]), m[2]
            else:
                m = re.search(r"\d+", s)
                if not m:
                    raise ValueError("看不出注水口，请用例如 8: 3 的格式")
                e, rest = int(m[0]), s[m.end():]
            rest = rest.strip()
            kw = ("没有", "无", "不出", "不流")
            if rest in ("", "-", "—", "0"):
                out = []
            elif not ints(rest) and any(k in rest for k in kw):
                out = []
            elif not ints(rest):
                raise ValueError("出水口要写数字，或用 - 表示不出水")
            else:
                out = ints(rest)
            if not 1 <= e <= n:
                raise ValueError(f"注水口必须在 1..{n} 之间")
            if any(not 1 <= x <= n for x in out):
                raise ValueError(f"出水口必须在 1..{n} 之间")
            if e in out:
                raise ValueError("注水口不能同时是出水口")
            if len(set(out)) < len(out):
                raise ValueError("出水口重复")
        except ValueError as ex:
            raise ValueError(f"第 {no} 行「{s}」：{ex}") from None
        ts.append((e, out))
    return ts

def build(n, wg, ts):
    """整理一轮：西岸组、东岸块、块大小。"""
    bs = []
    for e, out in ts:
        B = set([e, *out])
        hit = [i for i, b in enumerate(bs) if b & B]
        if not hit:
            bs.append(B)
        elif len(hit) > 1 or bs[hit[0]] != B:
            raise ValueError(
                f"东{e} 的观测与同轮其他观测冲突；"
                "请检查是否漏记/多记，或把不同组的数据混在一起"
            )

    bs = sorted((tuple(sorted(b)) for b in bs), key=lambda b: b[0])
    seen = {w for g in wg for w in g}
    wg = list(wg) + [[w] for w in range(1, n + 1) if w not in seen]
    wg.sort(key=min)

    wsize = Counter(len(g) for g in wg)
    bsize = Counter(len(b) for b in bs)
    for m, need in bsize.items():
        have = wsize.get(m, 0)
        if not have:
            raise ValueError(f"东岸出现了 {m} 根一组，但西岸没有 {m} 根相连的组")
        if need > have:
            raise ValueError(
                f"东岸有 {need} 个 {m} 根一组，但西岸只有 {have} 个 "
                f"{m} 根相连的组；请检查是否漏记/多记出水口"
            )

    wgof = [0] * (n + 1)
    for i, g in enumerate(wg):
        for w in g:
            wgof[w] = i
    eb = [-1] * (n + 1)
    for i, b in enumerate(bs):
        for e in b:
            eb[e] = i
    return wg, wgof, [len(g) for g in wg], eb, [len(b) for b in bs]

def solve(n, rounds, limit=50):
    """回溯枚举所有满足“东岸块 -> 西岸组”约束的双射。"""
    K = len(rounds)
    wgof = [r[1] for r in rounds]
    gs = [r[2] for r in rounds]
    eb = [r[3] for r in rounds]
    bs = [r[4] for r in rounds]
    a, used = [0] * (n + 1), [False] * (n + 1)
    # bg[k][b]：东岸块 b 认领的组；go[k][g]：-1 空闲，-2 未测端口，>=0 某块
    bg = [[-1] * len(bs[k]) for k in range(K)]
    go = [[-1] * len(gs[k]) for k in range(K)]
    sols = []

    def candidates(e):
        out = []
        for w in range(1, n + 1):
            if used[w]:
                continue
            for k in range(K):
                g, b = wgof[k][w], eb[k][e]
                if b < 0:
                    if go[k][g] >= 0:
                        break
                elif (
                    gs[k][g] != bs[k][b]
                    or bg[k][b] not in (-1, g)
                    or go[k][g] not in (-1, b)
                ):
                    break
            else:
                out.append(w)
        return out

    def dfs(depth):
        if len(sols) >= limit:
            return
        if depth == n:
            sols.append(tuple(a[1:]))
            return
        e, cand = min(
            ((e, candidates(e)) for e in range(1, n + 1) if not a[e]),
            key=lambda x: len(x[1]),
        )
        if not cand:
            return
        for w in cand:
            a[e], used[w] = w, True
            changed = []
            for k in range(K):
                g, b = wgof[k][w], eb[k][e]
                if b < 0:
                    if go[k][g] == -1:
                        go[k][g] = -2
                        changed.append((k, -1, g))
                elif bg[k][b] == -1:
                    bg[k][b], go[k][g] = g, b
                    changed.append((k, b, g))
            dfs(depth + 1)
            for k, b, g in reversed(changed):
                if b < 0:
                    go[k][g] = -1
                else:
                    bg[k][b], go[k][g] = -1, -1
            a[e], used[w] = 0, False

    dfs(0)
    return sols

def output(n, sols):
    if not sols:
        print("\n无解：这些观测不可能同时成立。")
        return
    for i, sol in enumerate(sols[:5], 1):
        w2e = [0] * (n + 1)
        for e, w in enumerate(sol, 1):
            w2e[w] = e
        print("\n唯一解：" if len(sols) == 1 else f"\n方案 {i}：")
        print("  西岸→东岸：", "  ".join(
            f"西{w}->东{w2e[w]}" for w in range(1, n + 1)
        ))
        print("  东岸→西岸：", "  ".join(
            f"东{e}->西{sol[e-1]}" for e in range(1, n + 1)
        ))
    if len(sols) > 5:
        print(f"\n……已找到 {len(sols)} 种可能（仅显示前 5 种）。")


def ask_n():
    """循环询问 N，输入非整数或过小时重问。"""
    while True:
        s = input("管道数量 N（至少 3）：").strip()
        if not s.isdigit():
            print("请输入一个整数，例如 10 或 50。")
            continue
        n = int(s)
        if n < 3:
            print("N 至少为 3。")
            continue
        return n


def read_round(n, pairs, i, total):
    """读一轮观测；任何输入错误都整轮重输。"""
    while True:
        print(f"\n第 {i}/{total} 轮：请在西岸连接")
        print("  " + describe(pairs, n))
        print("本轮注水/出水记录（每行一条，空行结束）：")
        try:
            ts = tests(n)
            r = build(n, pairs, ts)
            missing = [e for e in range(1, n + 1) if r[3][e] < 0]
            if missing:
                show = "、".join(f"东{e}" for e in missing[:8])
                more = "…" if len(missing) > 8 else ""
                raise ValueError(
                    f"还有 {show}{more} 没有出现在任何一组里；"
                    "请对每组至少测一根，并写全出水口"
                )
            return r
        except ValueError as ex:
            print(f"\n本轮输入有误：{ex}")
            print("提示：两两相连的组注水后应恰好有 1 个出水口；不连接的组应无水。")
            print("请重新输入本轮观测。")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    while True:
        try:
            n = ask_n()
            p1, p2 = article_groups(n)
            rounds = [
                read_round(n, p1, 1, 2),
                read_round(n, p2, 2, 2),
            ]
            sols = solve(n, rounds)
            output(n, sols)
            if sols:
                return 0

            print("可能是两轮记录互相矛盾，或实际接线与提示不一致。")
            if input("\n是否重新开始？[Y/n] ").strip().lower() in ("n", "no", "不", "否"):
                return 1
            print()
        except EOFError:
            print("\n已退出。")
            return 0
        except KeyboardInterrupt:
            print("\n已退出。")
            return 0
        except ValueError as ex:
            print(f"\n出错了：{ex}")
            if input("是否重新开始？[Y/n] ").strip().lower() in ("n", "no", "不", "否"):
                return 1
            print()


if __name__ == "__main__":
    sys.exit(main())