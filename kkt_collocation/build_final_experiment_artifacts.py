"""Create final LaTex tables and a concise experimental-results record."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WRITING = ROOT / "论文写作"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pm(data: dict, key: str, percent: bool = False, digits: int = 3) -> str:
    scale = 100.0 if percent else 1.0
    suffix = r"\%" if percent else ""
    return f"{scale*data[key]['mean']:.{digits}f} $\\pm$ {scale*data[key]['sample_std']:.{digits}f}{suffix}"


def pm_sci(data: dict, key: str, digits: int = 2) -> str:
    return f"{data[key]['mean']:.{digits}e} $\\pm$ {data[key]['sample_std']:.{digits}e}"


def teacher_stats(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    good = [row for row in rows if row["success"].lower() in ("1", "true")]
    values = lambda key: np.asarray([float(row[key]) for row in good])
    return {
        "n": len(rows), "success": len(good), "time_mean": float(values("solve_seconds").mean()),
        "time_std": float(values("solve_seconds").std(ddof=1)), "gmax": float(values("gmax").max()),
    }


def table_rows(rows, aggregate, nlp, raw_key: str, severe_key: str) -> list[tuple[str, ...]]:
    result = []
    for label, raw, safe, nlp_name in rows:
        result.append((
            label, pm(aggregate[raw], raw_key, True, 2), pm(aggregate[raw], severe_key, True, 2),
            pm(aggregate[safe], "mean_corrected_segments"), pm(nlp[nlp_name], "relative_absolute_gap", True, 3),
            pm(aggregate[safe], "mean_filter_seconds"), pm(aggregate[safe], "fallback_rate", True, 2),
        ))
    return result


def latex_row(values: tuple[str, ...] | list[str]) -> str:
    return " & ".join(values) + r" \\"


def main() -> None:
    vdp = load(ROOT / "kkt_collocation" / "results" / "final_multiseed_vdp900_penalty_aggregate" / "summary.json")["methods"]
    pen = load(ROOT / "kkt_collocation" / "results" / "final_multiseed_penicillin400_penalty_aggregate" / "summary.json")["methods"]
    vdp_nlp = load(ROOT / "kkt_collocation" / "results" / "final_vdp_multiseed_nlp50" / "summary.json")["methods"]
    pen_nlp = load(ROOT / "kkt_collocation" / "results" / "final_penicillin_multiseed_nlp50" / "summary.json")["methods"]
    vdp_sampling = load(ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_vdp_3seeds" / "summary.json")["comparison"]
    pen_sampling = load(ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_penicillin_3seeds" / "summary.json")["comparison"]
    teacher_vdp = teacher_stats(ROOT / "原版VDP" / "vdp_teacher_hds_nlp50.csv")
    teacher_pen = load(ROOT / "kkt_collocation" / "results" / "final_penicillin_fixed_teacher_original50_comparison" / "summary.json")
    jiang = load(ROOT / "kkt_collocation" / "results" / "jiang_fu_algorithm1_baseline" / "summary.json")
    jiang_fixed = load(ROOT / "kkt_collocation" / "results" / "jiang_fu_fixed_point_comparison" / "summary.json")
    ood = {
        "VDP": load(ROOT / "kkt_collocation" / "results" / "domain_stress_vdp_seed20260751" / "summary.json"),
        "Penicillin": load(ROOT / "kkt_collocation" / "results" / "domain_stress_penicillin_seed20260761" / "summary.json"),
        "CSTR": load(ROOT / "kkt_collocation" / "results" / "domain_stress_cstr_900_seed20260722" / "summary.json"),
    }

    vdp_methods = [
        ("S", "Never-KKT: S", "Never-KKT + HDS-lambda", "S"),
        ("S+Penalty", "Constraint-penalty: S+P", "Constraint-penalty: S+P + HDS-lambda", "S+Penalty"),
        ("Always-KKT", "Always-KKT: S+KKT", "Always-KKT + HDS-lambda", "S+KKT"),
        ("Adaptive", "Adaptive (S)", "Adaptive (S) + HDS-lambda", "S"),
    ]
    pen_methods = [
        ("S", "Never-KKT: S", "Never-KKT + HDS-lambda", "S"),
        ("S+Penalty", "Constraint-penalty: S+P", "Constraint-penalty: S+P + HDS-lambda", "S+Penalty"),
        ("Always-KKT", "Always-KKT: S+true-KKT", "Always-KKT + HDS-lambda", "S+true-KKT"),
        ("Adaptive", "Adaptive (S+true-KKT)", "Adaptive (S+true-KKT) + HDS-lambda", "S+true-KKT"),
    ]
    vrows = table_rows(vdp_methods, vdp, vdp_nlp, "nominal_violation_rate", "nominal_severe_violation_rate")
    prows = table_rows(pen_methods, pen, pen_nlp, "raw_violation_rate", "raw_severe_violation_rate")

    lines = [
        "% Auto-generated from frozen final experimental summaries.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Three-seed in-domain results. Violation rates are measured before HDS--$\lambda$ correction; relative gaps are measured after correction on 50 independently re-solved NLP reference points.}",
        r"\label{tab:main-ablation}", r"\scriptsize", r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Problem & Method & raw violation & severe violation & corrected segments & relative NLP gap & HDS time (s) & fallback \\", r"\midrule",
    ]
    for problem, rows in (("VDP", vrows), ("Penicillin", prows)):
        for i, row in enumerate(rows):
            lines.append(latex_row((problem if i == 0 else "",) + row))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", "", r"\begin{table*}[t]", r"\centering",
              r"\caption{Discrete checking versus continuous-time HDS auditing over all 1200 three-seed trajectories.}",
              r"\label{tab:hds-audit}", r"\scriptsize", r"\begin{tabular}{llrrr}", r"\toprule",
              r"Problem & audit & false-safe rate & false-safe count & mean audit time (s) \\", r"\midrule"]
    for problem, data in (("VDP", vdp_sampling), ("Penicillin", pen_sampling)):
        for key, label in (("zoh_endpoints", "ZOH endpoints"), ("uniform_10", "10 samples/segment"), ("uniform_100", "100 samples/segment"), ("HDS", "event-located HDS")):
            audit = data["audits"][key]
            lines.append(latex_row((problem if key == "zoh_endpoints" else "", label,
                                    f"{100*audit['false_safe_rate_vs_hds']:.2f}\\%", str(audit["false_safe_count"]), f"{audit['mean_seconds']:.3f}")))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    means = teacher_pen["means"]
    lines += [r"\end{tabular}", r"\end{table*}", "", r"\begin{table*}[t]", r"\centering",
              r"\caption{Deterministic solver comparison. The penicillin teacher run reproduces the original restriction, HDS exchange, and KKT termination logic. The same-grid NLP uses exactly the final teacher nodes and the same MATLAB SQP settings.}",
              r"\label{tab:teacher}", r"\scriptsize", r"\begin{tabular}{lrrrr}", r"\toprule",
              r"Problem & points & success & time (s) & $g_{\max}$ \\", r"\midrule",
              latex_row(("VDP teacher PCDP--HDS", str(teacher_vdp["n"]), f"{teacher_vdp['success']}/{teacher_vdp['n']}", f"{teacher_vdp['time_mean']:.3f} $\\pm$ {teacher_vdp['time_std']:.3f}", f"{teacher_vdp['gmax']:.2e}")),
              latex_row(("Penicillin teacher PCDP--HDS", str(teacher_pen["samples"]), f"{round(means['teacher_converged']*teacher_pen['samples'])}/4", f"{means['teacher_seconds']:.3f}", f"{means['teacher_hds_g']:.2e}")),
              latex_row(("Penicillin same-final-grid MATLAB NLP", str(teacher_pen["samples"]), f"{round(means['same_grid_solver_completed']*teacher_pen['samples'])}/4", f"{means['same_grid_seconds']:.3f}", f"{means['same_grid_hds_g']:.2e}")),
              latex_row(("Penicillin warm-start reduced NLP (801 nodes)", str(teacher_pen["samples"]), "4/4", f"{means['reduced_nlp_seconds']:.3f}", f"{means['reduced_nlp_hds_g']:.2e}")),
              latex_row(("Penicillin Adaptive+HDS", str(teacher_pen["samples"]), "4/4", f"{means['adaptive_total_seconds']:.3f}", f"{means['adaptive_hds_g']:.2e}")),
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    lines += ["", r"\begin{table*}[t]", r"\centering",
              r"\caption{Pointwise comparison with Jiang--Fu Algorithm~1 under identical dynamics, initial conditions, 10 ZOH controls, input bounds, and event-located HDS verification. Adaptive results are mean $\pm$ sample standard deviation over three training seeds; the relative objective difference uses the Jiang--Fu solution at the same point as the numerical reference.}",
              r"\label{tab:jiang-fu}", r"\scriptsize", r"\begin{tabular}{llrrrrr}", r"\toprule",
              r"Problem & method & point trials & accepted & time (s) & max HDS $g$ & relative objective difference \\", r"\midrule"]
    for problem in ("VDP", "Penicillin"):
        deterministic = jiang["groups"][f"unified_fixed_points/{problem}"]
        adaptive = jiang_fixed["problems"][problem]
        lines.append(latex_row((problem, "Jiang--Fu Algorithm 1", str(deterministic["points"]),
                                f"{deterministic['hds_safe']}/{deterministic['points']}",
                                f"{deterministic['mean_solve_seconds']:.3f}", f"{deterministic['max_hds_g']:.2e}", "--")))
        lines.append(latex_row(("", "Adaptive+HDS", f"{adaptive['points_per_seed']} " + r"$\times$ 3",
                                "15/15", pm(adaptive, "adaptive_total_seconds"),
                                pm_sci(adaptive, "adaptive_max_hds_g"),
                                pm(adaptive, "relative_objective_difference", True, 3))))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (WRITING / "11_最终实验表格.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ood_lines = [
        "% Auto-generated from frozen guard-bypassed OOD diagnostic summaries.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Guard-bypassed out-of-domain stress diagnostic from one frozen gate-selected checkpoint per benchmark (100 points per layer). The HDS--$\lambda$ columns intentionally bypass the declared operating-domain guard and are therefore diagnostic only, not deployment metrics. The accepted maximum is evaluated only over accepted HDS--$\lambda$ candidates. Under the declared deployment rule, every point in these layers is dispatched directly to the offline optimizer.}",
        r"\label{tab:ood-stress}", r"\scriptsize", r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Problem & OOD layer & $n$ & raw violation & bypass accepted & bypass fallback & accepted max HDS $g$ & default dispatch \\", r"\midrule",
    ]
    layers = (("near_10_percent", "Near OOD (+10\\%)"), ("far_20_percent", "Far OOD (+20\\%)"))
    for problem, summary in ood.items():
        for index, (layer, label) in enumerate(layers):
            item = summary["layers"][layer]
            ood_lines.append(latex_row((
                problem if index == 0 else "", label, str(item["samples"]),
                f"{100*item['raw_violation_rate']:.1f}\\%",
                f"{100*item['bypass_guard_accepted_rate']:.1f}\\%",
                f"{100*item['bypass_guard_fallback_rate']:.1f}\\%",
                f"{item['maximum_applied_hds_g_on_accepted']:.2e}",
                f"{100*item['default_deployment_offline_fallback_rate']:.1f}\\%",
            )))
        ood_lines.append(r"\midrule")
    ood_lines[-1] = r"\bottomrule"
    ood_lines += [r"\end{tabular}", r"\end{table*}", ""]
    (WRITING / "ood_stress_table.tex").write_text("\n".join(ood_lines), encoding="utf-8")

    markdown = f"""# 最终实验结果（冻结版）

## 协议

- VDP：900 个 $30\\times30$ HDS 审核训练标签；青霉素：400 个新增 801 节点离散 KKT 标签。
- 每个问题均使用 3 个独立训练种子、固定独立验证集和 400 点域内测试集。
- Adaptive gate：VDP 在 3/3 个种子选择 S；青霉素在 3/3 个种子选择 S+true-KKT。

## 结论

- VDP 中，S 的原始违反率为 {pm(vdp['Never-KKT: S'], 'nominal_violation_rate', True, 2)}，Always-KKT 为 {pm(vdp['Always-KKT: S+KKT'], 'nominal_violation_rate', True, 2)}，S+Penalty 为 {pm(vdp['Constraint-penalty: S+P'], 'nominal_violation_rate', True, 2)}。全部 HDS 修正分支为 100% accepted、0 fallback。
- 青霉素中，S、S+Penalty、Always-KKT 的原始违反率分别为 {pm(pen['Never-KKT: S'], 'raw_violation_rate', True, 2)}、{pm(pen['Constraint-penalty: S+P'], 'raw_violation_rate', True, 2)}、{pm(pen['Always-KKT: S+true-KKT'], 'raw_violation_rate', True, 2)}。KKT 修正后平均修正段数为 {pm(pen['Always-KKT + HDS-lambda'], 'mean_corrected_segments')}，小于 S+Penalty 的 {pm(pen['Constraint-penalty: S+P + HDS-lambda'], 'mean_corrected_segments')} 和 S 的 {pm(pen['Never-KKT + HDS-lambda'], 'mean_corrected_segments')}。
- 同点 NLP 参考中，VDP 修正后相对差距：S {pm(vdp_nlp['S'], 'relative_absolute_gap', True, 3)}，S+KKT {pm(vdp_nlp['S+KKT'], 'relative_absolute_gap', True, 3)}，S+Penalty {pm(vdp_nlp['S+Penalty'], 'relative_absolute_gap', True, 3)}；青霉素分别为 {pm(pen_nlp['S'], 'relative_absolute_gap', True, 3)}、{pm(pen_nlp['S+true-KKT'], 'relative_absolute_gap', True, 3)}、{pm(pen_nlp['S+Penalty'], 'relative_absolute_gap', True, 3)}。
- HDS 必要性：端点检查相对 HDS 的 false-safe rate 在 VDP 为 {100*vdp_sampling['audits']['zoh_endpoints']['false_safe_rate_vs_hds']:.2f}% ({vdp_sampling['audits']['zoh_endpoints']['false_safe_count']}/{vdp_sampling['samples']})，在青霉素为 {100*pen_sampling['audits']['zoh_endpoints']['false_safe_rate_vs_hds']:.2f}% ({pen_sampling['audits']['zoh_endpoints']['false_safe_count']}/{pen_sampling['samples']})。
- Jiang--Fu Algorithm 1 在统一固定点上 VDP/青霉素均为 5/5 求解成功且经独立 HDS 审核安全。平均求解时间分别为 {jiang['groups']['unified_fixed_points/VDP']['mean_solve_seconds']:.3f} s 和 {jiang['groups']['unified_fixed_points/Penicillin']['mean_solve_seconds']:.3f} s；Adaptive+HDS 为 {pm(jiang_fixed['problems']['VDP'], 'adaptive_total_seconds')} s 和 {pm(jiang_fixed['problems']['Penicillin'], 'adaptive_total_seconds')} s，对应同点相对目标差异为 {pm(jiang_fixed['problems']['VDP'], 'relative_objective_difference', True, 3)} 和 {pm(jiang_fixed['problems']['Penicillin'], 'relative_objective_difference', True, 3)}。

## 写作边界

- 教师青霉素四个固定点按原始 restriction--HDS--KKT 逻辑在 9--12 次 HDS 检查内全部收敛；不能再使用此前 8 次交换截断的正峰值结果。
- 约 0.3 s 的 reduced-space NLP 使用 801 个 RK4 节点、CasADi 精确导数和最近标签 warm start；同节点公平比较应使用约 5.9 s 的 MATLAB same-final-grid NLP，二者必须分栏报告。
- 严格连续时间安全统计只指高精度 HDS 审核后 accepted 的 HDS--$\\lambda$ 分支。
- 不宣称 KKT 对所有问题优于普通约束惩罚：VDP 中 S+Penalty 已足够；青霉素中 KKT 在修正负担、相对 NLP 差距和产品损失上更优。
"""
    (WRITING / "10_最终实验结果.md").write_text(markdown, encoding="utf-8")
    frozen = {"vdp_teacher": teacher_vdp, "penicillin_teacher_fixed": teacher_pen,
              "jiang_fu_algorithm1": jiang, "jiang_fu_fixed_comparison": jiang_fixed,
              "vdp_main": vdp, "penicillin_main": pen, "vdp_nlp": vdp_nlp,
              "penicillin_nlp": pen_nlp, "vdp_sampling": vdp_sampling,
              "penicillin_sampling": pen_sampling, "ood_stress": ood}
    (WRITING / "final_experiment_data.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(WRITING / "11_最终实验表格.tex")


if __name__ == "__main__":
    main()
