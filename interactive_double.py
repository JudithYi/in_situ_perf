#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
fs = 18
ub = 3.5

def async_min(f_sim, f_insitu1, f_insitu2, f_comm, fre, gpus, ppg):
    cores_min = gpus / 4
    cores_max = (gpus/4)*(18-ppg)
    p = np.arange(cores_min, cores_max+1, 1)
    t_total = np.zeros_like(p,dtype=float)
    for i in range(p.shape[0]):
        if f_sim(gpus,gpus*ppg) + fre * f_insitu1(gpus,gpus*ppg) > fre * f_insitu2(gpus,p[i]):
            t_total[i] = f_sim(gpus,gpus*ppg) + fre * f_insitu1(gpus,gpus*ppg) + fre *f_comm(gpus*ppg, p[i])
        else:
            t_total[i] = fre * f_insitu2(gpus,p[i]) + fre *f_comm(gpus*ppg, p[i])                                     
    return np.min(t_total), p[np.argmin(t_total)]

def interactive_auto(f_sim, f_insitu1, f_insitu2, f_comm, gpu, ppg, fre):
    t_hybrid, config = async_min(f_sim, f_insitu1, f_insitu2, f_comm, fre, gpu, ppg)
    t_sync = f_sim(gpu,gpu*ppg) + f_insitu1(gpu,gpu*ppg) * fre + f_insitu2(gpu,gpu*ppg) * fre
    def f_insitu(a,b):
        return f_insitu1(a,b) + f_insitu2(a,b)
    if t_hybrid < t_sync:
        print("Hybrid method is preferred.")
        config_str = "The best number of cores for in-situ task: " + str(config)
        print(config_str)
        interactive_hybrid(f_sim, f_insitu1, f_insitu2, f_comm, gpu, ppg, config, fre)
    else:
        print("Synchronous method is preferred.")
        interactive_sync(f_sim, f_insitu, gpu,ppg,fre)

def interactive_sync(f_sim, f_insitu, gpu, ppg, fre):
    g = np.arange(1, 51, 1)
    p = np.arange(1, 51*5, 1)
    t_nek = f_sim(g, g*ppg)
    t_catalyst = f_insitu(gpu, p) * fre
    t_nek_1 = f_sim(1, ppg)
    t_catalyst_1 = f_insitu(gpu, 1) * fre

    s_nek = t_nek_1 / t_nek
    s_catalyst = t_catalyst_1 / t_catalyst

    e_nek = s_nek / g
    e_catalyst = s_catalyst / p
    t_nek_ref = f_sim(gpu,gpu*ppg)
    t_catalyst_ref = f_insitu(gpu,gpu*ppg) * fre
    t_total = t_nek_ref + t_catalyst_ref
    ub = t_total 
    tmp_min = t_nek_1
    if tmp_min > t_catalyst_1:
        tmp_min = t_catalyst_1
    if tmp_min > t_total:
        ub = tmp_min
    ub = 1.1 * ub
    color=['#AC92EB', '#4FC1E8', '#A0D568', '#ED5564', '#FFCE54']
    color_2=['#C7395F','#E8BA40']
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=2, nrows=2,  figsize=(30*0.33,24*0.33))
    xticks = np.array(range(1,9))
    axs[0][0].plot(-g, t_nek, linewidth=2, color=color[0])
    axs[0][0].grid(True)
    axs[0][0].set_ylim(-0., ub)
    axs[0][1].set_ylim(-0., ub)
    axs[0][0].set_ylabel("Execution time (seconds)", fontsize=fs)
    axs[0][1].plot(p, t_catalyst, linewidth=2, color=color[2])
    axs[0][1].grid(True)
    axs[0][0].set_xlim(-51, 0)
    axs[0][1].set_xlim(0,251)
    axs[0][0].tick_params(right = False , labelbottom = False, bottom = False) 
    axs[0][1].tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    lines=[]
    line, =axs[1][0].plot(-g, -e_nek, linewidth=2, color=color[0])
    lines.append(line)
    line, =axs[1][1].plot(p, -e_catalyst, linewidth=2, color=color[2])
    lines.append(line)
    axs[1][0].grid(True)
    axs[1][0].set_xlabel("Number of GPUs for application", fontsize=fs)
    axs[1][0].set_ylabel("Efficiency", fontsize=fs)
    axs[1][1].grid(True)
    axs[1][1].set_xlabel("Number of cores for in-situ task", fontsize=fs)
    axs[1][1].set_ylim(-1.05, 0.0)
    axs[1][0].set_ylim(-1.05, 0.0)
    axs[1][0].set_xlim(-51, 0)
    axs[1][1].set_xlim(0,251)

    axs[1][0].set_yticks(-np.arange(0.2, 1.1, 0.2, dtype=float))
    y_ticks = []
    for i in range(1,6):
        y_ticks.append(str(float("{:.2f}".format(0.2*i))))
    axs[1][0].set_yticklabels(y_ticks, fontsize=fs)

    y_ref = np.arange(1, 2201, 10)/2000 - 0.05
    x_ref = y_ref - y_ref - gpu
    axs[1][0].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = y_ref - y_ref + gpu*ppg
    axs[1][1].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    #t_nek_ref = (0.064+2711.049/(core))
    #t_catalyst_ref = (3.806 + 8.248/np.sqrt(core))*fre

    x_ref = - np.arange(0, 2001, 10)/2000 * gpu
    y_ref = x_ref - x_ref + t_total
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = np.arange(0, 2001, 10)/2000 * gpu*ppg
    y_ref = x_ref - x_ref + t_total
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_nek_ref
    x_ref = y_ref - y_ref - gpu
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_catalyst_ref + t_nek_ref
    x_ref = y_ref - y_ref - gpu
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_catalyst_ref 
    x_ref = y_ref - y_ref + gpu * ppg 
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_nek_ref + t_catalyst_ref
    x_ref = y_ref - y_ref + gpu * ppg 
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = 'dashdot')
    axs[0][0].scatter(-gpu, t_total, marker="X", s = 100, color=color[3])
    tmp=axs[0][1].scatter(gpu*ppg, t_total, marker="X", s = 100, color=color[3])
    lines.append(tmp)
    axs[0][0].legend(lines,["NEKO", "Compression", "Total"], fontsize=fs,loc='upper left')
    lines = []
    tmp=axs[1][0].scatter(-gpu, -t_nek_1/t_nek_ref/gpu, marker="*", s = 100, color=color[0])
    lines.append(tmp)
    tmp=axs[1][1].scatter(gpu*ppg, -t_catalyst_1/t_catalyst_ref/(gpu*ppg), marker="*", s = 100, color=color[2])
    lines.append(tmp)
    axs[1][0].legend(lines,["Nek5000", "Compression"], fontsize=fs,loc='upper left')
    
    
    axs[1][1].tick_params(left = False, right = False , labelleft = False) 
    
    axs[0][0].set_xticks(-np.arange(0,51,5))
    axs[1][0].set_xticks(-np.arange(0,51,5))
    axs[1][0].set_xticklabels(np.arange(0,51,5), fontsize=fs)
    axs[0][1].set_xticks(np.arange(5,51,5)*5)
    axs[1][1].set_xticks(np.arange(5,51,5)*5)
    axs[1][1].set_xticklabels(np.arange(5,51,5)*5, fontsize=fs)
    if ub < 0.3:
        axs[0][0].set_yticks(np.arange(6)*0.002*int(ub*100))
        axs[0][1].set_yticks(np.arange(6)*0.002*int(ub*100))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.002*int(ub*100)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    elif ub < 3:
        axs[0][0].set_yticks(np.arange(6)*0.02*int(ub*10))
        axs[0][1].set_yticks(np.arange(6)*0.02*int(ub*10))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.02*int(ub*10)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    else:
        axs[0][0].set_yticks(np.arange(6)*0.2*int(ub*1))
        axs[0][1].set_yticks(np.arange(6)*0.2*int(ub*1))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.2*int(ub*1)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
 
def interactive_hybrid(f_sim, f_insitu1, f_insitu2, f_comm, gpu, ppg, core, fre):    
    g = np.arange(1, 51, 1)
    p = np.arange(1, 251, 1)
    t_nek = f_sim(g, g*ppg)
    t_catalyst1 = f_insitu1(gpu, p) * fre
    t_catalyst2 = f_insitu2(gpu, p) * fre
    t_nek_1 = f_sim(1, ppg)
    t_catalyst1_1 = f_insitu1(gpu, 1) * fre
    t_catalyst2_1 = f_insitu2(gpu, 1) * fre

    s_nek = t_nek_1 / t_nek
    s_catalyst1 = t_catalyst1_1 / t_catalyst1
    s_catalyst2 = t_catalyst2_1 / t_catalyst2

    e_nek = s_nek / g
    e_catalyst1 = s_catalyst1 / p
    e_catalyst2 = s_catalyst2 / p
    t_nek_ref = f_sim(gpu,gpu*ppg)
    t_catalyst1_ref = f_insitu1(gpu,gpu*ppg) * fre
    t_catalyst2_ref = f_insitu2(gpu,core) * fre
    t_catalyst2_1_ref = f_insitu2(gpu,gpu) * fre
    t_comm = f_comm(gpu*ppg, core) * fre
    if t_nek_ref + t_catalyst1_ref > t_catalyst2_ref :
        t_total = t_nek_ref + t_catalyst1_ref + t_comm
        color_idx = 0
        l_style = 'dashdot'
    else:
        t_total = t_catalyst2_ref + t_comm
        color_idx = 2
        l_style = 'dotted'
    if t_nek_ref + t_catalyst1_ref > t_catalyst2_1_ref :
        t_tota_ref_1 = t_nek_ref + t_catalyst1_ref + t_comm
    else:
        t_tota_ref_1 = t_catalyst2_1_ref + t_comm
    if t_tota_ref_1 > t_total:    
        ub = t_tota_ref_1 
    else:
        ub = t_total 
    tmp_min = t_nek_1
    if tmp_min > t_catalyst1_1:
        tmp_min = t_catalyst1_1
    if tmp_min > t_catalyst2_1:
        tmp_min = t_catalyst2_1
    if tmp_min > ub:
        ub = tmp_min
    ub = 1.1 * ub
    color=['#AC92EB', '#4FC1E8', '#A0D568', '#ED5564', '#FFCE54']
    color_2=['#C7395F','#E8BA40']
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=2, nrows=2,  figsize=(30*0.33,24*0.33))
    axs[0][0].plot(-g, t_nek, linewidth=2, color=color[0])
    axs[0][0].grid(True)
    axs[0][0].set_ylim(-0., ub)
    axs[0][1].set_ylim(-0., ub)
    axs[0][0].set_ylabel("Execution time (seconds)", fontsize=fs)
    axs[0][1].plot(p, t_catalyst1, linewidth=2, color=color[2])
    axs[0][1].plot(p, t_catalyst2, linewidth=2, color=color[2], linestyle = 'dashed')
    axs[0][1].grid(True)
    axs[0][0].set_xlim(-51, 0)
    axs[0][1].set_xlim(0,251)
    axs[0][0].tick_params(right = False , labelbottom = False, bottom = False) 
    axs[0][1].tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    lines=[]
    line, =axs[1][0].plot(-g, -e_nek, linewidth=2, color=color[0])
    lines.append(line)
    line, =axs[1][1].plot(p, -e_catalyst1, linewidth=2, color=color[2])
    lines.append(line)
    line, =axs[1][1].plot(p, -e_catalyst2, linewidth=2, color=color[2], linestyle = 'dashed')
    lines.append(line)
    axs[1][0].grid(True)
    axs[1][0].set_xlabel("Number of GPUs for application", fontsize=fs)
    axs[1][0].set_ylabel("Efficiency", fontsize=fs)
    axs[1][1].grid(True)
    axs[1][1].set_xlabel("Number of cores for in-situ task", fontsize=fs)
    axs[1][1].set_ylim(-1.05, 0.0)
    axs[1][0].set_ylim(-1.05, 0.0)
    axs[1][0].set_xlim(-51, 0)
    axs[1][1].set_xlim(0,251)

    axs[1][0].set_yticks(-np.arange(0.2, 1.1, 0.2, dtype=float))
    y_ticks = []
    for i in range(1,6):
        y_ticks.append(str(float("{:.2f}".format(0.2*i))))
    axs[1][0].set_yticklabels(y_ticks, fontsize=fs)
    y_ref = np.arange(1, 2201, 10)/2000 - 0.05
    x_ref = y_ref - y_ref - gpu
    axs[1][0].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = y_ref - y_ref + gpu*ppg
    axs[1][1].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = y_ref - y_ref + core
    axs[1][1].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dotted')

    x_ref = - np.arange(0, 2001, 10)/2000 * gpu
    y_ref = x_ref - x_ref + t_total
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = l_style)
    x_ref = np.arange(0, 2001, 10)/2000 * core
    y_ref = x_ref - x_ref + t_total
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = l_style)
    y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_comm + (t_total - t_comm)
    x_ref = y_ref - y_ref - gpu
    tmp2, = axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[4], linestyle = l_style)
    x_ref = y_ref - y_ref + core
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[4], linestyle = l_style)
    lines.append(tmp2)
    if color_idx > 1:
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_catalyst2_ref
        x_ref = y_ref - y_ref - gpu
        axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[color_idx], linestyle = l_style)
        x_ref = y_ref - y_ref + core
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[color_idx], linestyle = l_style)
    else:
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_nek_ref
        x_ref = y_ref - y_ref - gpu
        axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = l_style)
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_catalyst1_ref + t_nek_ref
        x_ref = y_ref - y_ref - gpu
        axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = l_style)
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_catalyst1_ref
        x_ref = y_ref - y_ref + core
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = l_style)
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_nek_ref + t_catalyst1_ref
        x_ref = y_ref - y_ref + core
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = l_style)

    tmp=axs[0][0].scatter(-gpu, t_total, marker="X", s = 100, color=color[3])
    lines.append(tmp)
    axs[0][1].scatter(core, t_total, marker="X", s = 100, color=color[3])
    axs[0][0].legend(lines,["NEKO", "Lossy compression", "Lossless compression", "Communication", "Total"], fontsize=fs, loc='upper left')
    lines=[]
    tmp=axs[1][0].scatter(-gpu, -t_nek_1/t_nek_ref/gpu, marker="*", s = 100, color=color[0])
    lines.append(tmp)
    tmp=axs[1][1].scatter(gpu*ppg, -t_catalyst1_1/t_catalyst1_ref/(gpu*ppg), marker="*", s = 100, color=color[2])
    lines.append(tmp)
    tmp=axs[1][1].scatter(core, -t_catalyst2_1/t_catalyst2_ref/core, marker="o", s = 100, color=color[2])
    lines.append(tmp)
    axs[1][0].legend(lines,["NEKO", "Lossy compression", "Lossless compression"], fontsize=fs,loc='upper left')

    axs[1][1].tick_params(left = False, right = False , labelleft = False) 
    
    axs[0][0].set_xticks(-np.arange(0,51,5))
    axs[1][0].set_xticks(-np.arange(0,51,5))
    axs[1][0].set_xticklabels(np.arange(0,51,5), fontsize=fs)
    axs[0][1].set_xticks(np.arange(5,51,5)*5)
    axs[1][1].set_xticks(np.arange(5,51,5)*5)
    axs[1][1].set_xticklabels(np.arange(5,51,5)*5, fontsize=fs)
    if ub < 0.3:
        axs[0][0].set_yticks(np.arange(6)*0.002*int(ub*100))
        axs[0][1].set_yticks(np.arange(6)*0.002*int(ub*100))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.002*int(ub*100)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    elif ub < 3:
        axs[0][0].set_yticks(np.arange(6)*0.02*int(ub*10))
        axs[0][1].set_yticks(np.arange(6)*0.02*int(ub*10))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.02*int(ub*10)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    else:
        axs[0][0].set_yticks(np.arange(6)*0.2*int(ub*1))
        axs[0][1].set_yticks(np.arange(6)*0.2*int(ub*1))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.2*int(ub*1)))))
        axs[0][0].set_yticklabels(y_ticks,fontsize=fs)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def interactive_insitu(gpu, ppg, core, freq, method): 
    fre = float(1.0/freq)
    if method == "Synchronous":
        interactive_sync(f_neko, comp, gpu, ppg, fre)
    elif method == "Hybrid":
        interactive_hybrid(f_neko, lossy, lossless, f_adios, gpu, ppg, core, fre)
    elif method == "Auto":
        interactive_auto(f_neko, lossy, lossless, f_adios, gpu, ppg, fre)

def f_nek(p):
    return (0.064+2711.049/(p))

def f_img(p):
    return (3.806 + 8.248/np.sqrt(p))

def f_adios(p1, p2):
    return f_adios_insitumpi(5368709120,p1,p2)

def f_adios_insitumpi(message_size, p1, p2):
    return message_size/(2**30)/(0.21809503032972155 + 0.03678039198016967 * np.log2(p1) * np.log2(p2) * np.log2(message_size) + 0.4794909512524979 * np.log2(p1) + 0.04892764040096403 * np.log2(message_size))

def f_neko(g,p):
    return 2.6042 * 10 ** (-2) + 1.78424 / g

def lossy (g,p):
    return 2.5573 * 10 ** (-3) + 2.4793 * 10 ** (-2) / g + 0.86584 / p

def lossless(g, p):
    return 0.1384 + 55.3399 / p

def comp(g, p):
    return lossy(g,p) + lossless(g,p)