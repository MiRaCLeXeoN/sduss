import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

if not os.path.exists("./results.csv"):
    raise FileNotFoundError("results.csv not found. Please run `python scripts/draw/get_metric.py` first.")

results = pd.read_csv("./results.csv")

distrifusion_c=0
nirvana_c=1
mixcache_c=2
mixfusion_c=3

label_names = ["Distrifusion", "Nirvana", "Mix-Cache", "PatchedServe"]

c = [
    ("#f47f72", "#C72228", ),
    ("#B8E5FA", "#0C4E9B", ),
    ("#EEC186", "#FF991C", ),
    ("#B2DBB9", "#156434", ),
]

colors = [
    *[combo[0] for combo in c],
    "#EEF0A7", 
    "#F7A6AC", 
    "#F7B7D2",
]
dark_colors = [
    *[combo[1] for combo in c],
    "#FFBC80",
    "#6B98C4", 
    "#F5867F", 
]

line_markers = [
    "s", "o", "D", "X", "*",
]

left_label_color = "black"
right_label_color = "black"
yticklabel_fontsize = 16
xticklabel_fontsize = 16
label_fontsize = 18

bar_alpha = 0.7
line_width = 3
legend_fontsize = 16

def get_data(model, qps, policy, slo, dp_size):
    if policy == "mixfusion":
        policy = "esymred"
    elif policy == "mixed_cache":
        policy = "fcfs_mixed"
    elif policy == "nirvana":
        policy = "orca_resbyres"
    elif policy == "distrifusion":
        policy = "distrifusion"
    else:
        raise ValueError(f"Unknown policy: {policy}. Cannot convert to any proper name.")
    
    if isinstance(qps, float):
        qps = f"{qps:.1f}"

    d = results[
        (results["model"] == model) & 
        (results["qps"] == qps) & 
        (results["policy"] == policy) & 
        (results["slo"] == slo) & 
        (results["dp_size"] == dp_size)
    ]
    assert len(d) >= 1, f"Data corrupted for {model}, {qps}, {policy}, {slo}, {dp_size}, num rows={len(d)}"
    slo_rate = d["slo_rate"].values[0]
    goodput = d["goodput"].values[0]
    return slo_rate, goodput


def figure_12():
    plt.clf()
    # categories = ['A', 'B', 'C', 'D', 'E']
    sdxl_qps = [0.8, 0.9, 1.0, 1.1, 1.2]
    sd3_qps = [0.1, 0.2, 0.3, 0.4, 0.5]

    # sdxl
    # sdxl_nirvana_slo = [0.386, 0.258, 0.244, 0.198, 0.144]
    # sdxl_mixed_cache_slo = [1, 0.998, 0.994, 0.986, 0.926]
    # sdxl_mixfusion_slo = [1, 1, 1, 0.998, 0.93]
    sdxl_nirvana_slo = [d[0] for d in [get_data("sdxl", qps, "nirvana", 5, dp_size=1) for qps in sdxl_qps]]
    sdxl_mixed_cache_slo = [d[0] for d in [get_data("sdxl", qps, "mixed_cache", 5, dp_size=1) for qps in sdxl_qps]]
    sdxl_mixfusion_slo = [d[0] for d in [get_data("sdxl", qps, "mixfusion", 5, dp_size=1) for qps in sdxl_qps]]

    # sd3
    # sd3_nirvana_slo = [1, 0.964, 0.83, 0.584, 0.228]
    # sd3_mixed_cache_slo = [1, 0.994, 0.974, 0.868, 0.566]
    # sd3_mixfusion_slo = [1, 1, 0.998, 0.982, 0.868]
    sd3_nirvana_slo = [d[0] for d in [get_data("sd3", qps, "nirvana", 5, dp_size=1) for qps in sd3_qps]]
    sd3_mixed_cache_slo = [d[0] for d in [get_data("sd3", qps, "mixed_cache", 5, dp_size=1) for qps in sd3_qps]]
    sd3_mixfusion_slo = [d[0] for d in [get_data("sd3", qps, "mixfusion", 5, dp_size=1) for qps in sd3_qps]]

    # sdxl
    # sdxl_nirvana_goodput = [0.285, 0.196, 0.194, 0.157, 0.116]
    # sdxl_mixed_cache_goodput = [0.781, 0.892, 0.986, 1.06, 1.04]
    # sdxl_mixfusion_goodput = [0.781, 0.892, 1, 1.06, 1.04]
    sdxl_nirvana_goodput = [d[1] for d in [get_data("sdxl", qps, "nirvana", 5, dp_size=1) for qps in sdxl_qps]]
    sdxl_mixed_cache_goodput = [d[1] for d in [get_data("sdxl", qps, "mixed_cache", 5, dp_size=1) for qps in sdxl_qps]]
    sdxl_mixfusion_goodput = [d[1] for d in [get_data("sdxl", qps, "mixfusion", 5, dp_size=1) for qps in sdxl_qps]]

    # sd3
    # sd3_nirvana_goodput = [0.109, 0.194, 0.266, 0.223, 0.095]
    # sd3_mixed_cache_goodput = [0.109, 0.2, 0.312, 0.338, 0.269]
    # sd3_mixfusion_goodput = [0.109, 0.201, 0.32, 0.38, 0.412]
    sd3_nirvana_goodput = [d[1] for d in [get_data("sd3", qps, "nirvana", 5, dp_size=1) for qps in sd3_qps]]
    sd3_mixed_cache_goodput = [d[1] for d in [get_data("sd3", qps, "mixed_cache", 5, dp_size=1) for qps in sd3_qps]]
    sd3_mixfusion_goodput = [d[1] for d in [get_data("sd3", qps, "mixfusion", 5, dp_size=1) for qps in sd3_qps]]

    slo_x_axis = np.arange(5)

    # 设置阈值
    threshold = 90

    # 创建图和3个子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # 第一个子图
    ax1 = axes[0]
    ax1_twin = ax1.twinx() # 创建共享x轴的第二个Y轴
    bar_width = 0.3
    # 绘制柱状图
    ax1.bar(slo_x_axis, np.array(sdxl_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width, np.array(sdxl_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha,  edgecolor='black', label=f'{label_names[mixcache_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width * 2, np.array(sdxl_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')
    ax1.set_ylabel('SLO Satisfaction (%)', fontsize=label_fontsize)
    ax1.set_xlabel('QPS (Req/s)\nSDXL', fontsize=label_fontsize)
    ax1.tick_params(axis='y', labelcolor=left_label_color, labelsize=yticklabel_fontsize)
    ax1.set_ylim(0, 100) # 柱状图Y轴范围
    ax1.set_xticks(slo_x_axis + 1 * bar_width, sdxl_qps, fontsize=xticklabel_fontsize)
    yticks = list(ax1.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f} Threshold' # 找到阈值并修改其标签
    ax1.set_yticks(yticks)

    # 绘制折线图
    ax1_twin.plot(slo_x_axis + 1 * bar_width, sdxl_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1 * bar_width, sdxl_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1 * bar_width, sdxl_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')
    # ax1_twin.set_ylabel('Goodput (Req/s)', fontsize=18)
    ax1_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=yticklabel_fontsize)
    # ax1_twin.set_ylim(0, 150) # 折线图Y轴范围

    # 添加阈值线
    ax1.axhline(y=threshold, color='gray', linestyle='--')

    # 设置标题和图例
    # ax1.set_title('SDXL')
    handles_bar, labels_bar = fig.sca(ax1).get_legend_handles_labels()
    handles_plot, labels_plot = fig.sca(ax1_twin).get_legend_handles_labels()
    handles = []
    labels = []
    for index in range(len(handles_bar)):
        handles.append(handles_bar[index])
        handles.append(handles_plot[index])
        labels.append(labels_bar[index])
        labels.append(labels_plot[index])


    # 第二个子图
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.bar(slo_x_axis, np.array(sd3_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width, np.array(sd3_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width * 2, np.array(sd3_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')

    # ax2.set_ylabel('SLO Satisfaction (%)', fontsize=18)
    ax2.tick_params(axis='y', labelcolor=left_label_color, labelsize=yticklabel_fontsize)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('QPS (Req/s)\nSD3', fontsize=label_fontsize)


    ax2_twin.plot(slo_x_axis + 1 * bar_width, sd3_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1 * bar_width, sd3_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1 * bar_width, sd3_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')

    ax2_twin.set_ylabel('Goodput (Req/s)', fontsize=label_fontsize)
    ax2_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=yticklabel_fontsize)
    # ax2_twin.set_ylim(0, 120)
    ax2.set_xticks(slo_x_axis + 1 * bar_width, sd3_qps, fontsize=xticklabel_fontsize)
    ax2.axhline(y=threshold, color='green', linestyle='--')
    # ax2.set_title('SD3', fontsize=20)

    yticks = list(ax2.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f}' # 找到阈值并修改其标签
    ax2.set_yticks(yticks)

    # plt.subplots_adjust(bottom=0.01, top=0.75, left=0.05, right=0.95)
    # plt.tight_layout(rect=[0.03, 0, 0.97, 0.9])
    # plt.savefig("e2e_performance.pdf", dpi=500, format="pdf")
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(2.5, 1.5), ncol=3, fontsize=legend_fontsize, bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig("e2e_performance.pdf", dpi=500, format="pdf", bbox_inches="tight")

def figure_13():
    plt.clf()
    my_xticklabel_fontsize = 18
    my_yticklabel_fontsize = 18
    my_label_fontsize = 20

    sdxl_qps_list = ["8.8_small", "8.8_medium", "8.8_large"]
    sd3_qps_list = ["3.2_small", "3.2_medium", "3.2_large"]
    # sdxl
    # sdxl_distrifusion_slo = [0.026, 0.026, 0.026]
    # sdxl_nirvana_slo = [0.492, 0.504, 0.464]
    # sdxl_mixed_cache_slo = [1, 0.994, 0.764]
    # sdxl_mixfusion_slo = [1, 0.994, 0.768]
    sdxl_distrifusion_slo = [d[0] for d in [get_data("sdxl", qps, "distrifusion", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_nirvana_slo = [d[0] for d in [get_data("sdxl", qps, "nirvana", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_mixed_cache_slo = [d[0] for d in [get_data("sdxl", qps, "mixed_cache", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_mixfusion_slo = [d[0] for d in [get_data("sdxl", qps, "mixfusion", 5, dp_size=8) for qps in sdxl_qps_list]]

    # sd3
    # sd3_distrifusion_slo = [0.018, 0.018, 0.006]
    # sd3_nirvana_slo = [0.996, 0.872, 0.634]
    # sd3_mixed_cache_slo = [1, 0.988, 0.846]
    # sd3_mixfusion_slo = [1, 0.99, 0.96]
    sd3_distrifusion_slo = [d[0] for d in [get_data("sd3", qps, "distrifusion", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_nirvana_slo = [d[0] for d in [get_data("sd3", qps, "nirvana", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_mixed_cache_slo = [d[0] for d in [get_data("sd3", qps, "mixed_cache", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_mixfusion_slo = [d[0] for d in [get_data("sd3", qps, "mixfusion", 5, dp_size=8) for qps in sd3_qps_list]]

    # sdxl
    # sdxl_distrifusion_goodput = [0.051, 0.049, 0.044]
    # sdxl_nirvana_goodput = [2.69, 2.73, 2.34]
    # sdxl_mixed_cache_goodput = [7.46, 7.3, 4.86]
    # sdxl_mixfusion_goodput = [7.46, 7.3, 4.96]
    sdxl_distrifusion_goodput = [d[1] for d in [get_data("sdxl", qps, "distrifusion", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_nirvana_goodput = [d[1] for d in [get_data("sdxl", qps, "nirvana", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_mixed_cache_goodput = [d[1] for d in [get_data("sdxl", qps, "mixed_cache", 5, dp_size=8) for qps in sdxl_qps_list]]
    sdxl_mixfusion_goodput = [d[1] for d in [get_data("sdxl", qps, "mixfusion", 5, dp_size=8) for qps in sdxl_qps_list]]

    # sd3
    # sd3_distrifusion_goodput = [0.014, 0.013, 0.003]
    # sd3_nirvana_goodput = [3.14, 2.7, 1.7]
    # sd3_mixed_cache_goodput = [3.14, 3.06, 2.47]
    # sd3_mixfusion_goodput = [3.15, 3.07, 2.72]
    sd3_distrifusion_goodput = [d[1] for d in [get_data("sd3", qps, "distrifusion", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_nirvana_goodput = [d[1] for d in [get_data("sd3", qps, "nirvana", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_mixed_cache_goodput = [d[1] for d in [get_data("sd3", qps, "mixed_cache", 5, dp_size=8) for qps in sd3_qps_list]]
    sd3_mixfusion_goodput = [d[1] for d in [get_data("sd3", qps, "mixfusion", 5, dp_size=8) for qps in sd3_qps_list]]

    slo_x_axis = np.arange(3)

    distribution = ["Low", "Medium", "High"]
    # 设置阈值
    threshold = 90

    # 创建图和3个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 第一个子图
    ax1 = axes[0]
    ax1_twin = ax1.twinx() # 创建共享x轴的第二个Y轴
    bar_width = 0.2
    # 绘制柱状图
    ax1.bar(slo_x_axis, np.array(sdxl_distrifusion_slo) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width, np.array(sdxl_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width * 2, np.array(sdxl_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha,edgecolor='black', label=f'{label_names[mixcache_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width * 3, np.array(sdxl_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha,edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')
    ax1.set_ylabel('SLO Satisfaction\n(%)', fontsize=my_label_fontsize)
    ax1.set_xlabel('Distribution\nSDXL', fontsize=my_label_fontsize)
    # ax1.tick_params(axis='y', labelcolor='mediumpurple')
    ax1.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
    ax1.set_ylim(0, 100) # 柱状图Y轴范围
    ax1.set_xticks(slo_x_axis + 1.5 * bar_width, distribution, fontsize=my_xticklabel_fontsize)
    yticks = list(ax1.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f} Threshold' # 找到阈值并修改其标签
    ax1.set_yticks(yticks)

    # 绘制折线图
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_distrifusion_goodput, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')
    # ax1_twin.set_ylabel('Goodput (Req/s)', fontsize=18)
    # ax1_twin.tick_params(axis='y', labelcolor='lightcoral')
    ax1_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
    # ax1_twin.set_ylim(0, 150) # 折线图Y轴范围

    # 添加阈值线
    ax1.axhline(y=threshold, color='green', linestyle='--')

    # 设置标题和图例
    # ax1.set_title('SDXL')
    handles_bar, labels_bar = fig.sca(ax1).get_legend_handles_labels()
    handles_plot, labels_plot = fig.sca(ax1_twin).get_legend_handles_labels()
    handles = []
    labels = []
    for index in range(len(handles_bar)):
        handles.append(handles_bar[index])
        handles.append(handles_plot[index])
        labels.append(labels_bar[index])
        labels.append(labels_plot[index])



    # 第二个子图
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.bar(slo_x_axis, np.array(sd3_distrifusion_slo) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width, np.array(sd3_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width * 2, np.array(sd3_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
    ax2.bar(slo_x_axis + bar_width * 3, np.array(sd3_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')

    ax2.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Distribution\nSD3', fontsize=my_label_fontsize)


    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_distrifusion_goodput, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c],linewidth=line_width,  label=f'{label_names[nirvana_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')

    ax2_twin.set_ylabel('Goodput\n(Req/s)', fontsize=my_label_fontsize)
    # ax2_twin.tick_params(axis='y', labelcolor='lightcoral')
    ax2_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
    # ax2_twin.set_ylim(0, 120)
    ax2.set_xticks(slo_x_axis + 1 * bar_width, distribution, fontsize=my_xticklabel_fontsize)

    ax2.axhline(y=threshold, color='green', linestyle='--')
    # ax2.set_title('SD3', fontsize=20)

    yticks = list(ax2.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f}' # 找到阈值并修改其标签
    ax2.set_yticks(yticks)

    # plt.subplots_adjust(bottom=0.22, top=0.8, left=0.05, right=0.95)
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.88])
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(2.6, 1.4), ncol=4, fontsize=legend_fontsize, bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig("sensitivity_distribution.pdf", dpi=500, format="pdf", bbox_inches="tight")

def figure_14():
    plt.clf()
    slo_x_axis = np.arange(5)
    # 设置阈值
    threshold = 90
    my_xticklabel_fontsize = 18
    my_yticklabel_fontsize = 18
    my_label_fontsize = 20

    def plot(axes, model_name):
        # if model_name == "SDXL":
        #     # sdxl
        #     distrifusion_slo_2 = [0.096, 0.052, 0.044, 0.038, 0.038]
        #     nirvana_slo_2 = [0.476, 0.36, 0.322, 0.296, 0.256]
        #     mixed_cache_slo_2 = [1, 1, 1, 0.984, 0.508]
        #     mixfusion_slo_2 = [1, 1, 1, 0.986, 0.722]

        #     distrifusion_goodput_2 = [0.102, 0.054, 0.047, 0.041, 0.04]
        #     nirvana_goodput_2 = [0.664, 0.537, 0.48, 0.448, 0.38]
        #     mixed_cache_goodput_2 = [1.6, 1.79, 1.93, 2.05, 1.13]
        #     mixfusion_goodput_2 = [1.6, 1.79, 1.93, 2.05, 1.71]

        #     # mixfusion gpu4 qps=0.8 rerun
        #     distrifusion_slo_4 = [0.034, 0.032, 0.032, 0.03, 0.026]
        #     nirvana_slo_4 = [0.532, 0.456, 0.418, 0.278, 0.324]
        #     mixed_cache_slo_4 = [1, 0.988, 0.98, 0.516, 0.428]
        #     mixfusion_slo_4 = [1, 0.994, 0.99, 0.708, 0.656]

        #     distrifusion_goodput_4 = [0.053, 0.05, 0.05, 0.047, 0.04]
        #     nirvana_goodput_4 = [1.38, 1.26, 1.18, 0.81, 0.95]
        #     mixed_cache_goodput_4 = [3.03, 3.36, 3.55, 2.0, 1.68]
        #     mixfusion_goodput_4 = [3.03, 3.39, 3.62, 2.74, 2.85]


        #     distrifusion_slo_8 = [0.038, 0.036, 0.034, 0.028, 0.026]
        #     nirvana_slo_8 = [0.57, 0.516, 0.458, 0.438, 0.362]
        #     mixed_cache_slo_8 = [1, 1, 1, 0.936, 0.788]
        #     mixfusion_slo_8 = [1, 1, 1, 0.95, 0.802]

        #     distrifusion_goodput_8 = [0.074, 0.07, 0.06, 0.054, 0.05]
        #     nirvana_goodput_8 = [2.79, 2.65, 2.36, 2.35, 1.99]
        #     mixed_cache_goodput_8 = [5.88, 6.25, 6.84, 6.5, 5.96]
        #     mixfusion_goodput_8 = [5.88, 6.25, 6.84, 6.58, 5.98]
        #     qps = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        # elif model_name == "SD3":
        #     # sd3
        #     distrifusion_slo_2 = [0.988, 0.652, 0.042, 0.02, 0.018]
        #     nirvana_slo_2 = [1, 0.99, 0.962, 0.612, 0.346]
        #     mixed_cache_slo_2 = [1, 1, 0.996, 0.9, 0.778]
        #     mixfusion_slo_2 = [1, 1, 1, 0.97, 0.922]

        #     distrifusion_goodput_2 = [0.198, 0.251, 0.018, 0.008, 0.008]
        #     nirvana_goodput_2 = [0.201, 0.389, 0.549, 0.484, 0.279]
        #     mixed_cache_goodput_2 = [0.201, 0.392, 0.567, 0.722, 0.723]
        #     mixfusion_goodput_2 = [0.201, 0.392, 0.569, 0.777, 0.842]


        #     distrifusion_slo_4 = [0.928, 0.044, 0.022, 0.02, 0.01]
        #     nirvana_slo_4 = [1, 0.998, 0.962, 0.788, 0.512]
        #     mixed_cache_slo_4 = [1, 1, 0.994, 0.904, 0.532]
        #     mixfusion_slo_4 = [1, 1, 0.996, 0.982, 0.898]

        #     distrifusion_goodput_4 = [0.361, 0.026, 0.013, 0.011, 0.006]
        #     nirvana_goodput_4 = [0.393, 0.807, 1.16, 1.17, 0.815]
        #     mixed_cache_goodput_4 = [0.393, 0.807, 1.19, 1.35, 0.960]
        #     mixfusion_goodput_4 = [0.393, 0.807, 1.19, 1.46, 1.62]


        #     distrifusion_slo_8 = [0.064, 0.024, 0.016, 0.016, 0.014]
        #     nirvana_slo_8 = [1, 1, 0.994, 0.808, 0.666]
        #     mixed_cache_slo_8 = [1, 1, 1, 0.96, 0.804]
        #     mixfusion_slo_8 = [1, 1, 1, 0.98, 0.912]

        #     distrifusion_goodput_8 = [0.044, 0.016, 0.011, 0.011, 0.009]
        #     nirvana_goodput_8 = [0.809, 1.53, 2.4, 2.34, 1.97]
        #     mixed_cache_goodput_8 = [0.809, 1.53, 2.4, 2.82, 2.75]
        #     mixfusion_goodput_8 = [0.809, 1.53, 2.4, 2.88, 2.98]

        sdxl_qps = [0.8, 0.9, 1.0, 1.1, 1.2]
        sd3_qps = [0.1, 0.2, 0.3, 0.4, 0.5]

        if model_name == "SDXL":
            qps_list = sdxl_qps
            model = "sdxl"
        elif model_name == "SD3":
            qps_list = sd3_qps
            model = "sd3"
        
        data = {}
        for dp_size in [2,4,8]:
            for policy in ["distrifusion", "nirvana", "mixed_cache", "mixfusion"]:
                slo_key = f"{policy}_slo_{dp_size}"
                goodput_key = f"{policy}_goodput_{dp_size}"
                dp_qps_list = [q * dp_size for q in qps_list]
                loaded_data = [get_data(model, qps, policy, 5, dp_size=dp_size) for qps in dp_qps_list]
                data[slo_key] = [d[0] for d in loaded_data]
                data[goodput_key] = [d[1] for d in loaded_data]
        
        distrifusion_slo_2 = data["distrifusion_slo_2"]
        nirvana_slo_2 = data["nirvana_slo_2"]
        mixed_cache_slo_2 = data["mixed_cache_slo_2"]
        mixfusion_slo_2 = data["mixfusion_slo_2"]

        distrifusion_goodput_2 = data["distrifusion_goodput_2"]
        nirvana_goodput_2 = data["nirvana_goodput_2"]
        mixed_cache_goodput_2 = data["mixed_cache_goodput_2"]
        mixfusion_goodput_2 = data["mixfusion_goodput_2"]

        distrifusion_slo_4 = data["distrifusion_slo_4"]
        nirvana_slo_4 = data["nirvana_slo_4"]
        mixed_cache_slo_4 = data["mixed_cache_slo_4"]
        mixfusion_slo_4 = data["mixfusion_slo_4"]

        distrifusion_goodput_4 = data["distrifusion_goodput_4"]
        nirvana_goodput_4 = data["nirvana_goodput_4"]
        mixed_cache_goodput_4 = data["mixed_cache_goodput_4"]
        mixfusion_goodput_4 = data["mixfusion_goodput_4"]

        distrifusion_slo_8 = data["distrifusion_slo_8"]
        nirvana_slo_8 = data["nirvana_slo_8"]
        mixed_cache_slo_8 = data["mixed_cache_slo_8"]
        mixfusion_slo_8 = data["mixfusion_slo_8"]

        distrifusion_goodput_8 = data["distrifusion_goodput_8"]
        nirvana_goodput_8 = data["nirvana_goodput_8"]
        mixed_cache_goodput_8 = data["mixed_cache_goodput_8"]
        mixfusion_goodput_8 = data["mixfusion_goodput_8"]
        
        if model_name == "SDXL":
            qps = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        elif model_name == "SD3":
            qps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # 第一个子图
        ax1 = axes[0]
        ax1_twin = ax1.twinx() # 创建共享x轴的第二个Y轴
        bar_width = 0.23
        # 绘制柱状图
        ax1.bar(slo_x_axis, np.array(distrifusion_slo_2) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
        ax1.bar(slo_x_axis + bar_width, np.array(nirvana_slo_2) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
        ax1.bar(slo_x_axis + bar_width * 2, np.array(mixed_cache_slo_2) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
        ax1.bar(slo_x_axis + bar_width * 3, np.array(mixfusion_slo_2) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')
        ax1.set_ylabel('SLO Satisfaction(%)', fontsize=my_label_fontsize)
        ax1.set_xlabel(f'QPS (Req/s)\n{model_name}-2 GPU', fontsize=my_label_fontsize)
        # ax1.tick_params(axis='y', labelcolor='mediumpurple')
        ax1.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
        ax1.set_ylim(0, 100) # 柱状图Y轴范围
        ax1.set_xticks(slo_x_axis + 1.5 * bar_width, qps * 2, fontsize=my_xticklabel_fontsize)
        yticks = list(ax1.get_yticks()) # 获取现有刻度
        if threshold not in yticks:
            yticks.append(threshold) # 添加阈值到刻度列表
        yticks.sort() # 排序
        yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
        if threshold in yticks:
            yticklabels[yticks.index(threshold)] = f'{threshold:.1f} Threshold' # 找到阈值并修改其标签
        ax1.set_yticks(yticks)

        # 绘制折线图
        ax1_twin.plot(slo_x_axis + 1.5 * bar_width, distrifusion_goodput_2, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
        ax1_twin.plot(slo_x_axis + 1.5 * bar_width, nirvana_goodput_2, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
        ax1_twin.plot(slo_x_axis + 1.5 * bar_width, mixed_cache_goodput_2, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
        ax1_twin.plot(slo_x_axis + 1.5 * bar_width, mixfusion_goodput_2, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')
        # ax1_twin.set_ylabel('Goodput (Req/s)', fontsize=15)
        # ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
        # ax1_twin.set_ylim(0, 150) # 折线图Y轴范围

        # 添加阈值线
        ax1.axhline(y=threshold, color='green', linestyle='--')

        # 第二个子图
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        ax2.bar(slo_x_axis, np.array(distrifusion_slo_4) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
        ax2.bar(slo_x_axis + bar_width, np.array(nirvana_slo_4) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
        ax2.bar(slo_x_axis + bar_width * 2, np.array(mixed_cache_slo_4) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
        ax2.bar(slo_x_axis + bar_width * 3, np.array(mixfusion_slo_4) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')

        # ax2.set_ylabel('SLO Satisfaction (%)', fontsize=15)
        # ax2.tick_params(axis='y', labelcolor='mediumpurple')
        ax2.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel(f'QPS (Req/s)\n{model_name}-4 GPU', fontsize=my_label_fontsize)

        ax2_twin.plot(slo_x_axis + 1.5 * bar_width, distrifusion_goodput_4, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
        ax2_twin.plot(slo_x_axis + 1.5 * bar_width, nirvana_goodput_4, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
        ax2_twin.plot(slo_x_axis + 1.5 * bar_width, mixed_cache_goodput_4, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
        ax2_twin.plot(slo_x_axis + 1.5 * bar_width, mixfusion_goodput_4, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')

        # ax2_twin.set_ylabel('Goodput (Req/s)', fontsize=15)
        # ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
        # ax2_twin.set_ylim(0, 120)
        ax2.set_xticks(slo_x_axis + 1.5 * bar_width, qps * 4, fontsize=my_xticklabel_fontsize)
        ax2.axhline(y=threshold, color='green', linestyle='--')
        # ax2.set_title('SD3', fontsize=20)

        yticks = list(ax2.get_yticks()) # 获取现有刻度
        if threshold not in yticks:
            yticks.append(threshold) # 添加阈值到刻度列表
        yticks.sort() # 排序
        yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
        if threshold in yticks:
            yticklabels[yticks.index(threshold)] = f'{threshold:.1f}' # 找到阈值并修改其标签
        ax2.set_yticks(yticks)

        # 第三个子图
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        ax3.bar(slo_x_axis, np.array(distrifusion_slo_8) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
        ax3.bar(slo_x_axis + bar_width, np.array(nirvana_slo_8) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
        ax3.bar(slo_x_axis + bar_width * 2, np.array(mixed_cache_slo_8) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
        ax3.bar(slo_x_axis + bar_width * 3, np.array(mixfusion_slo_8) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')

        # ax3.set_ylabel('SLO Satisfaction (%)', fontsize=15)
        # ax3.tick_params(axis='y', labelcolor='mediumpurple')
        ax3.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel(f'QPS (Req/s)\n{model_name}-8 GPU', fontsize=my_label_fontsize)

        ax3_twin.plot(slo_x_axis + 1.5 * bar_width, distrifusion_goodput_8, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
        ax3_twin.plot(slo_x_axis + 1.5 * bar_width, nirvana_goodput_8, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
        ax3_twin.plot(slo_x_axis + 1.5 * bar_width, mixed_cache_goodput_8, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
        ax3_twin.plot(slo_x_axis + 1.5 * bar_width, mixfusion_goodput_8, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')

        ax3_twin.set_ylabel('Goodput(Req/s)', fontsize=my_label_fontsize)
        # ax3_twin.tick_params(axis='y', labelcolor='red')
        ax3_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
        # ax2_twin.set_ylim(0, 120)
        ax3.set_xticks(slo_x_axis + 1.5 * bar_width, qps * 8, fontsize=my_xticklabel_fontsize)
        ax3.axhline(y=threshold, color='green', linestyle='--')
        # ax2.set_title('SD3', fontsize=20)

        yticks = list(ax3.get_yticks()) # 获取现有刻度
        if threshold not in yticks:
            yticks.append(threshold) # 添加阈值到刻度列表
        yticks.sort() # 排序
        yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
        if threshold in yticks:
            yticklabels[yticks.index(threshold)] = f'{threshold:.1f}' # 找到阈值并修改其标签
        ax3.set_yticks(yticks)

        return ax1_twin

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 6.8))

    ax1 = axes[0][0]
    ax1_twin = plot(axes[0], "SDXL")
    plot(axes[1], "SD3")

    # 设置标题和图例
    # ax1.set_title('SDXL')
    handles_bar, labels_bar = fig.sca(ax1).get_legend_handles_labels()
    handles_plot, labels_plot = fig.sca(ax1_twin).get_legend_handles_labels()
    handles = []
    labels = []
    for index in range(len(handles_bar)):
        handles.append(handles_bar[index])
        handles.append(handles_plot[index])
        labels.append(labels_bar[index])
        labels.append(labels_plot[index])

    print(handles, labels)
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(3.9, 1.6), ncol=4, fontsize=legend_fontsize, bbox_transform=ax1.transAxes)

    # plt.subplots_adjust(bottom=0.22, top=0.8, left=0.05, right=0.95)
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.85])
    plt.tight_layout()
    plt.savefig(f"sensitivity_gpu_num.pdf", dpi=500, format="pdf", bbox_inches="tight")

def figure_15():
    plt.clf()
    my_xticklabel_fontsize = 18
    my_yticklabel_fontsize = 18
    my_label_fontsize = 20

    slo_list = [3, 5, 10]

    data ={}
    for model in ["sdxl", "sd3"]:
        qps = 8.8 if model == "sdxl" else 3.2
        for policy in ["distrifusion", "nirvana", "mixed_cache", "mixfusion"]:
            slo_key = f"{model}_{policy}_slo"
            goodput_key = f"{model}_{policy}_goodput"
            loaded_data = [get_data(model, qps, policy, slo, dp_size=8) for slo in slo_list]
            data[slo_key] = [d[0] for d in loaded_data]
            data[goodput_key] = [d[1] for d in loaded_data]

    # sdxl
    sdxl_distrifusion_slo = data["sdxl_distrifusion_slo"]
    sdxl_nirvana_slo = data["sdxl_nirvana_slo"]
    sdxl_mixed_cache_slo = data["sdxl_mixed_cache_slo"]
    sdxl_mixfusion_slo = data["sdxl_mixfusion_slo"]

    # sd3
    sd3_distrifusion_slo = data["sd3_distrifusion_slo"]
    sd3_nirvana_slo = data["sd3_nirvana_slo"]
    sd3_mixed_cache_slo = data["sd3_mixed_cache_slo"]
    sd3_mixfusion_slo = data["sd3_mixfusion_slo"]

    # sdxl
    sdxl_distrifusion_goodput = data["sdxl_distrifusion_goodput"]
    sdxl_nirvana_goodput = data["sdxl_nirvana_goodput"]
    sdxl_mixed_cache_goodput = data["sdxl_mixed_cache_goodput"]
    sdxl_mixfusion_goodput = data["sdxl_mixfusion_goodput"]

    # sd3
    sd3_distrifusion_goodput = data["sd3_distrifusion_goodput"]
    sd3_nirvana_goodput = data["sd3_nirvana_goodput"]
    sd3_mixed_cache_goodput = data["sd3_mixed_cache_goodput"]
    sd3_mixfusion_goodput = data["sd3_mixfusion_goodput"]


    slo_x_axis = np.arange(3)
    SLO = ["3x", "5x", "10x"]
    # 设置阈值
    threshold = 90

    # 创建图和3个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 第一个子图
    ax1 = axes[0]
    ax1_twin = ax1.twinx() # 创建共享x轴的第二个Y轴
    bar_width = 0.2
    # 绘制柱状图
    ax1.bar(slo_x_axis, np.array(sdxl_distrifusion_slo) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width, np.array(sdxl_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax1.bar(slo_x_axis + bar_width * 2, np.array(sdxl_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
    ax1.bar(slo_x_axis + bar_width * 3, np.array(sdxl_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')
    ax1.set_ylabel('SLO Satisfaction\n(%)', fontsize=my_label_fontsize)
    ax1.set_xlabel('SLO Scale\nSDXL', fontsize=my_label_fontsize)
    # ax1.tick_params(axis='y', labelcolor='mediumpurple')
    ax1.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
    ax1.set_ylim(0, 100) # 柱状图Y轴范围
    ax1.set_xticks(slo_x_axis + 1.5 * bar_width, SLO, fontsize=my_xticklabel_fontsize)
    yticks = list(ax1.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f} Threshold' # 找到阈值并修改其标签
    ax1.set_yticks(yticks)

    # 绘制折线图
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_distrifusion_goodput, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax1_twin.plot(slo_x_axis + 1.5 * bar_width, sdxl_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')
    # ax1_twin.set_ylabel('Goodput (Req/s)', fontsize=18)
    # ax1_twin.tick_params(axis='y', labelcolor='lightcoral')
    ax1_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
    # ax1_twin.set_ylim(0, 150) # 折线图Y轴范围

    # 添加阈值线
    ax1.axhline(y=threshold, color='green', linestyle='--')

    # 设置标题和图例
    # ax1.set_title('SDXL')
    handles_bar, labels_bar = fig.sca(ax1).get_legend_handles_labels()
    handles_plot, labels_plot = fig.sca(ax1_twin).get_legend_handles_labels()
    handles = []
    labels = []
    for index in range(len(handles_bar)):
        handles.append(handles_bar[index])
        handles.append(handles_plot[index])
        labels.append(labels_bar[index])
        labels.append(labels_plot[index])

    print(handles, labels)



    # 第二个子图
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(slo_x_axis, np.array(sd3_distrifusion_slo) * 100, bar_width, color=colors[distrifusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[distrifusion_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width, np.array(sd3_nirvana_slo) * 100, bar_width, color=colors[nirvana_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[nirvana_c]}-SLO')
    ax2.bar(slo_x_axis + bar_width * 2, np.array(sd3_mixed_cache_slo) * 100, bar_width, color=colors[mixcache_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixcache_c]}-Cache-SLO')
    ax2.bar(slo_x_axis + bar_width * 3, np.array(sd3_mixfusion_slo) * 100, bar_width, color=colors[mixfusion_c], alpha=bar_alpha, edgecolor='black', label=f'{label_names[mixfusion_c]}-SLO')

    # ax2.set_ylabel('SLO Satisfaction (%)', fontsize=18)
    # ax2.tick_params(axis='y', labelcolor='mediumpurple')
    ax2.tick_params(axis='y', labelcolor=left_label_color, labelsize=my_yticklabel_fontsize)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('SLO Scale\nSD3', fontsize=my_label_fontsize)

    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_distrifusion_goodput, color=dark_colors[distrifusion_c], marker=line_markers[distrifusion_c], linewidth=line_width, label=f'{label_names[distrifusion_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_nirvana_goodput, color=dark_colors[nirvana_c], marker=line_markers[nirvana_c], linewidth=line_width, label=f'{label_names[nirvana_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_mixed_cache_goodput, color=dark_colors[mixcache_c], marker=line_markers[mixcache_c], linewidth=line_width, label=f'{label_names[mixcache_c]}-Goodput')
    ax2_twin.plot(slo_x_axis + 1.5 * bar_width, sd3_mixfusion_goodput, color=dark_colors[mixfusion_c], marker=line_markers[mixfusion_c], linewidth=line_width, label=f'{label_names[mixfusion_c]}-Goodput')

    ax2_twin.set_ylabel('Goodput\n(Req/s)', fontsize=my_label_fontsize)
    ax2_twin.tick_params(axis='y', labelcolor='lightcoral')
    ax2_twin.tick_params(axis='y', labelcolor=right_label_color, labelsize=my_yticklabel_fontsize)
    # ax2_twin.set_ylim(0, 120)
    ax2.set_xticks(slo_x_axis + 1.5 * bar_width, SLO, fontsize=my_xticklabel_fontsize)
    ax2.axhline(y=threshold, color='green', linestyle='--')
    # ax2.set_title('SD3', fontsize=20)

    yticks = list(ax2.get_yticks()) # 获取现有刻度
    if threshold not in yticks:
        yticks.append(threshold) # 添加阈值到刻度列表
    yticks.sort() # 排序
    yticklabels = [f'{y:.1f}' for y in yticks] # 格式化为字符串
    if threshold in yticks:
        yticklabels[yticks.index(threshold)] = f'{threshold:.1f}' # 找到阈值并修改其标签
    ax2.set_yticks(yticks)

    # plt.subplots_adjust(bottom=0.22, top=0.8, left=0.05, right=0.95)
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.88])
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(2.6, 1.4), ncol=4, fontsize=legend_fontsize, bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig("sensitivity_slo.pdf", dpi=500, format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    figure_12()
    figure_13()
    figure_14()
    figure_15()