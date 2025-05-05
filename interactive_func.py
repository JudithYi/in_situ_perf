#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
fs = 14 

def f_nek(p):
    return (0.064+2711.049/(p))

def f_img(p):
    return (3.806 + 8.248/np.sqrt(p))

def f_adios(p1, p2):
    return f_adios_insitumpi(5368709120,p1,p2)

def f_adios_insitumpi(message_size, p1, p2):
    return message_size/(2**30)/(0.21809503032972155 + 0.03678039198016967 * np.log2(p1) * np.log2(p2) * np.log2(message_size) + 0.4794909512524979 * np.log2(p1) + 0.04892764040096403 * np.log2(message_size))

def async_min(f_sim, f_insitu, f_comm, fre, cores):
    p = np.arange(1, cores,1)
    t_total = np.zeros_like(p,dtype=float)
    for i in range(p.shape[0]):
        if f_sim(p[i]) > fre * f_insitu(cores-p[i]):
            t_total[i] = f_sim(p[i]) + f_comm(p[i], cores-p[i])
        else:
            t_total[i] = fre * f_insitu(cores-p[i]) + f_comm(p[i], cores-p[i])                                       
    return np.min(t_total), np.argmin(t_total)
        

def _plot_sync_async(f_sim, f_insitu, f_comm, fre, nodes):
    step = int((nodes[1]-nodes[0])/10) 
    if step == 0:
        step = 1
    step = 72
    p = np.arange(nodes[0], nodes[1]+step, step)
    t_sim = f_sim(p)
    t_insitu = f_insitu(p)

    color=['#AC92EB', '#4FC1E8', '#A0D568', '#ED5564', '#FFCE54']
    color_2=['#C7395F','#E8BA40']
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=2, nrows=1,  figsize=(30*0.5,12*0.5))
    t_sync = t_sim + fre*t_insitu
    p_size = p.shape
    t_async = np.zeros(p_size[0],dtype=float)
    config_async = np.zeros(p_size[0],dtype=int)
    for i in range(p_size[0]):
        t_async[i], config_async[i] = async_min(f_sim, f_insitu, f_comm, fre, p[i])
    lines=[]
    line,=axs[0].plot(p, t_sync, linewidth=2, color=color[1])
    lines.append(line)
    line=axs[0].scatter(p, t_async, marker="X", s = 100, color=color[3])
    for i, txt in enumerate(config_async):
        axs[0].annotate(txt, (p[i], t_async[i]*1.1))
    lines.append(line)
    axs[0].grid(True)
    axs[0].set_ylabel("Execution time (seconds)", fontsize=fs)
    axs[0].set_xlabel("Number of cores", fontsize=fs)
    axs[0].set_xticks(p[::2])
    axs[0].set_xticklabels(p[::2], fontsize = fs)
    axs[0].legend(lines,["Sync", "Best async"], fontsize=fs)
    ub = np.max(t_sync)
    if ub < np.max(t_async):
        ub = np.max(t_async)
    ub = 1.1 * ub
    step = int((nodes[1]-np.min(config_async))/10)
    p_e = np.arange(int(np.min(config_async)),nodes[1]+step,step)
    
    t_sim = f_sim(p_e)
    t_insitu = f_insitu(p_e)
    
    t_sim_1 = f_sim(1)
    t_insitu_1 = f_insitu(1)

    s_sim = t_sim_1 / t_sim
    s_insitu = t_insitu_1 / t_insitu

    e_sim = s_sim / p_e
    e_insitu = s_insitu / p_e
    lines=[]
    line, =axs[1].plot(p_e, e_sim, linewidth=2, color=color[0])
    lines.append(line)
    line, =axs[1].plot(p_e, e_insitu, linewidth=2, color=color[2])
    lines.append(line)

    axs[1].grid(True)
    axs[1].set_xlabel("Number of cores", fontsize=fs)
    axs[1].set_ylabel("Efficiency", fontsize=fs)
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_yticks(np.arange(0, 1.1, 0.2, dtype=float))
    axs[1].set_xticks(p_e[::2])
    axs[1].set_xticklabels(p_e[::2], fontsize = fs)
    axs[1].legend(lines,["Nek5000", "Image generation"], fontsize=fs)
    y_ticks = []
    for i in range(6):
        y_ticks.append(str(float("{:.2f}".format(0.2*i))))
    axs[1].set_yticklabels(y_ticks, fontsize=fs)
    if ub < 0.3:
        axs[0].set_yticks(np.arange(6)*0.002*int(ub*100))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.002*int(ub*100)))))
        axs[0].set_yticklabels(y_ticks,fontsize=fs)
    elif ub < 3:
        axs[0].set_yticks(np.arange(6)*0.02*int(ub*10))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.02*int(ub*10)))))
        axs[0].set_yticklabels(y_ticks,fontsize=fs)
    else:
        axs[0].set_yticks(np.arange(6)*0.2*int(ub*1))
        y_ticks = []
        for i in range(6):
            y_ticks.append(str(float("{:.2f}".format(i*0.2*int(ub*1)))))
        axs[0].set_yticklabels(y_ticks,fontsize=fs)
    plt.tight_layout()
    plt.show()

def plot_sync_async(fre, nodes):
    _plot_sync_async(f_nek, f_img, f_adios, float(1/fre), nodes)
    