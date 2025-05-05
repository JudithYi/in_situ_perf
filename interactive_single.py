#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
fs = 18
def interactive_auto(f_sim, f_insitu, f_comm, total_core, fre):
    t_async_min, config_async_min = async_min(f_sim, f_insitu, f_comm, fre, total_core)
    t_sync = f_sim(total_core) + f_insitu(total_core) * fre
    if t_async_min < t_sync:
        print("Asynchronous approach is preferred.")
        config_str = "The best number of cores for application: " + str(config_async_min)
        print(config_str)
        interactive_async(f_sim, f_insitu, f_comm, total_core, config_async_min, fre)
    else:
        print("Synchronous approach is preferred.")
        interactive_sync(f_sim, f_insitu, total_core, fre)

def interactive_async(f_sim, f_insitu, f_comm, total_core, app_core, fre):
    sim_core=app_core
    p = np.arange(1, 2001, 10)
    #t_nek = (0.064+2711.049/(p))
    #t_catalyst = (3.806 + 8.248/np.sqrt(p)) * fre
    t_nek = f_sim(p)
    t_catalyst = f_insitu(p) * fre
    t_nek_1 = f_sim(1)
    t_catalyst_1 = f_insitu(1) * fre

    s_nek = t_nek_1 / t_nek
    s_catalyst = t_catalyst_1 / t_catalyst

    e_nek = s_nek / p
    e_catalyst = s_catalyst / p

    color=['#AC92EB', '#4FC1E8', '#A0D568', '#ED5564', '#FFCE54']
    color_2=['#C7395F','#E8BA40']
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=2, nrows=2,  figsize=(30*0.33,24*0.33))
    xticks = np.array(range(1,9))
    axs[0][0].plot(-p, t_nek, linewidth=2, color=color[0])
    axs[0][0].grid(True)
    axs[0][0].set_ylim(-0., 8.5)
    axs[0][1].set_ylim(-0., 8.5)
    axs[0][0].set_ylabel("Execution time (seconds)", fontsize=fs)
    axs[0][1].plot(p, t_catalyst, linewidth=2, color=color[2])
    axs[0][1].grid(True)
    axs[0][0].set_xlim(-2025, 0)
    axs[0][1].set_xlim(0,2025)
    axs[0][0].tick_params(right = False , labelbottom = False, bottom = False) 
    axs[0][1].tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    lines=[]
    line, =axs[1][0].plot(-p, -e_nek, linewidth=2, color=color[0])
    lines.append(line)
    line, =axs[1][1].plot(p, -e_catalyst, linewidth=2, color=color[2])
    lines.append(line)
    axs[1][0].grid(True)
    axs[1][0].set_xlabel("Number of cores for application", fontsize=fs)
    axs[1][0].set_ylabel("Efficiency", fontsize=fs)
    axs[1][1].grid(True)
    axs[1][1].set_xlabel("Number of cores for in-situ task", fontsize=fs)
    axs[1][1].set_ylim(-1.05, 0.0)
    axs[1][0].set_ylim(-1.05, 0.0)
    axs[1][0].set_xlim(-2025, 0)
    axs[1][1].set_xlim(0,2025)


    axs[1][0].set_xticks(-np.arange(0,2001,250))
    axs[1][0].set_xticklabels(np.arange(0,2001,250), fontsize=fs)
    
    axs[1][0].set_xticks(-np.arange(0,2001,250))
    axs[1][0].set_xticklabels(np.arange(0,2001,250), fontsize=fs)
    axs[1][0].set_yticks(-np.arange(0.2, 1.1, 0.2, dtype=float))
    y_ticks = []
    for i in range(1,6):
        y_ticks.append(str(float("{:.2f}".format(0.2*i))))
    axs[1][0].set_yticklabels(y_ticks, fontsize=fs)

    if(sim_core < total_core):
        y_ref = np.arange(1, 2201, 10)/2000 - 0.05
        x_ref = y_ref - y_ref - sim_core
        axs[1][0].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
        x_ref = y_ref - y_ref + total_core-sim_core
        axs[1][1].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
        #t_nek_ref = (0.064+2711.049/(sim_core))
        t_nek_ref = f_sim(sim_core)
        #t_catalyst_ref = (3.806 + 8.248/np.sqrt(total_core-sim_core))*fre
        t_catalyst_ref = f_insitu(total_core-sim_core)*fre
        t_comm = f_comm(sim_core, total_core-sim_core)*fre
        if(t_nek_ref > t_catalyst_ref):
            t_total = t_nek_ref + t_comm
            color_idx = 0
        else:
            t_total = t_catalyst_ref + t_comm
            color_idx = 2
        x_ref = - np.arange(0, 2001, 10)/2000 * sim_core
        y_ref = x_ref - x_ref + t_total
        axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
        x_ref = np.arange(0, 2001, 10)/2000 * (total_core - sim_core)
        y_ref = x_ref - x_ref + t_total
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
        y_ref = np.arange(0, 2001, 10)/2000 * (t_total - t_comm)
        x_ref = y_ref - y_ref - sim_core
        axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[color_idx], linestyle = 'dashdot')
        y_ref = np.arange(0, 2001, 10)/2000 * (t_total - t_comm)
        x_ref = y_ref - y_ref + total_core - sim_core
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[color_idx], linestyle = 'dashdot')
        axs[0][0].scatter(-sim_core, t_total, marker="X", s = 100, color=color[3])
        tmp=axs[0][1].scatter(total_core-sim_core, t_total, marker="X", s = 100, color=color[3])
        y_ref = np.arange(0.0,1.0,0.005,dtype=float) * t_comm + (t_total - t_comm)
        x_ref = y_ref - y_ref - sim_core
        tmp2, = axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[4], linestyle = 'dashdot')
        x_ref = y_ref - y_ref + total_core - sim_core
        axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[4], linestyle = 'dashdot')
        lines.append(tmp2)
        lines.append(tmp)
        axs[0][0].legend(lines,["Nek5000", "Image generation", "Communication", "Total"], fontsize=fs)
        lines = []
        tmp=axs[1][0].scatter(-sim_core, -t_nek_1/t_nek_ref/sim_core, marker="P", s = 100, color=color[0])
        lines.append(tmp)
        tmp=axs[1][1].scatter(total_core - sim_core, -t_catalyst_1/t_catalyst_ref/(total_core - sim_core), marker="P", s = 100, color=color[2])
        lines.append(tmp)
        axs[1][0].legend(lines,["Nek5000", "Image generation"], fontsize=fs,loc='upper left')


    else:
        axs[0][0].legend(lines,["Nek5000", "Image generation"], fontsize=fs)
            
    
    axs[1][1].tick_params(left = False, right = False , labelleft = False) 
    
        
    axs[0][0].set_yticks(np.arange(9))
    axs[0][0].set_yticklabels(np.arange(9),fontsize=fs)
    axs[1][1].set_xticks(np.arange(250,2001,250))
    axs[1][1].set_xticklabels(np.arange(250,2001,250), fontsize=fs)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
def interactive_sync(f_sim, f_insitu, core, fre):
    p = np.arange(1, 2001, 10)
    t_nek = f_sim(p)
    t_catalyst = f_insitu(p) * fre
    t_nek_1 = f_sim(1)
    t_catalyst_1 = f_insitu(1) * fre
    #t_nek = (0.064+2711.049/(p))
    #t_catalyst = (3.806 + 8.248/np.sqrt(p)) * fre
    #t_nek_1 = (0.064+2711.049)
    #t_catalyst_1 = (3.806 + 8.248/np.sqrt(1)) * fre

    s_nek = t_nek_1 / t_nek
    s_catalyst = t_catalyst_1 / t_catalyst

    e_nek = s_nek / p
    e_catalyst = s_catalyst / p

    color=['#AC92EB', '#4FC1E8', '#A0D568', '#ED5564', '#FFCE54']
    color_2=['#C7395F','#E8BA40']
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=2, nrows=2,  figsize=(30*0.33,24*0.33))
    xticks = np.array(range(1,9))
    axs[0][0].plot(-p, t_nek, linewidth=2, color=color[0])
    axs[0][0].grid(True)
    axs[0][0].set_ylim(-0., 8.5)
    axs[0][1].set_ylim(-0., 8.5)
    axs[0][0].set_ylabel("Execution time (seconds)", fontsize=fs)
    axs[0][1].plot(p, t_catalyst, linewidth=2, color=color[2])
    axs[0][1].grid(True)
    axs[0][0].set_xlim(-2025, 0)
    axs[0][1].set_xlim(0,2025)
    axs[0][0].tick_params(right = False , labelbottom = False, bottom = False) 
    axs[0][1].tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    lines=[]
    line, =axs[1][0].plot(-p, -e_nek, linewidth=2, color=color[0])
    lines.append(line)
    line, =axs[1][1].plot(p, -e_catalyst, linewidth=2, color=color[2])
    lines.append(line)
    axs[1][0].grid(True)
    axs[1][0].set_xlabel("Number of cores for application", fontsize=fs)
    axs[1][0].set_ylabel("Efficiency", fontsize=fs)
    axs[1][1].grid(True)
    axs[1][1].set_xlabel("Number of cores for in-situ task", fontsize=fs)
    axs[1][1].set_ylim(-1.05, 0.0)
    axs[1][0].set_ylim(-1.05, 0.0)
    axs[1][0].set_xlim(-2025, 0)
    axs[1][1].set_xlim(0,2025)


    axs[1][0].set_xticks(-np.arange(0,2001,250))
    axs[1][0].set_xticklabels(np.arange(0,2001,250), fontsize=fs)
    axs[1][0].set_yticks(-np.arange(0.2, 1.1, 0.2, dtype=float))
    y_ticks = []
    for i in range(1,6):
        y_ticks.append(str(float("{:.2f}".format(0.2*i))))
    axs[1][0].set_yticklabels(y_ticks, fontsize=fs)

    y_ref = np.arange(1, 2201, 10)/2000 - 0.05
    x_ref = y_ref - y_ref - core
    axs[1][0].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = y_ref - y_ref + core
    axs[1][1].plot(x_ref,-y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    #t_nek_ref = (0.064+2711.049/(core))
    #t_catalyst_ref = (3.806 + 8.248/np.sqrt(core))*fre
    t_nek_ref = f_sim(core)
    t_catalyst_ref = f_insitu(core) * fre
    t_total = t_nek_ref + t_catalyst_ref
    x_ref = - np.arange(0, 2001, 10)/2000 * core
    y_ref = x_ref - x_ref + t_total
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    x_ref = np.arange(0, 2001, 10)/2000 * core
    y_ref = x_ref - x_ref + t_total
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[3], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_nek_ref
    x_ref = y_ref - y_ref - core
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_catalyst_ref + t_nek_ref
    x_ref = y_ref - y_ref - core
    axs[0][0].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_catalyst_ref 
    x_ref = y_ref - y_ref + core 
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[2], linestyle = 'dashdot')
    y_ref = np.arange(0, 2001, 10)/2000 * t_nek_ref + t_catalyst_ref
    x_ref = y_ref - y_ref + core 
    axs[0][1].plot(x_ref,y_ref, linewidth=2, color=color[0], linestyle = 'dashdot')
    axs[0][0].scatter(-core, t_total, marker="X", s = 100, color=color[3])
    tmp=axs[0][1].scatter(core, t_total, marker="X", s = 100, color=color[3])
    lines.append(tmp)
    axs[0][0].legend(lines,["Nek5000", "Image generation", "Total"], fontsize=fs,loc='upper left')
    lines = []
    tmp=axs[1][0].scatter(-core, -t_nek_1/t_nek_ref/core, marker="P", s = 100, color=color[0])
    lines.append(tmp)
    tmp=axs[1][1].scatter(core, -t_catalyst_1/t_catalyst_ref/core, marker="P", s = 100, color=color[2])
    lines.append(tmp)
    axs[1][0].legend(lines,["Nek5000", "Image generation"], fontsize=fs,loc='upper left')
    
    
    axs[1][1].tick_params(left = False, right = False , labelleft = False) 
    
    axs[0][0].set_yticks(np.arange(9))
    axs[0][0].set_yticklabels(np.arange(9),fontsize=fs)
    axs[1][1].set_xticks(np.arange(250,2001,250))
    axs[1][1].set_xticklabels(np.arange(250,2001,250), fontsize=fs)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def async_min(f_sim, f_insitu, f_comm, fre, cores):
    p = np.arange(1, cores,1)
    t_total = np.zeros_like(p,dtype=float)
    for i in range(p.shape[0]):
        if f_sim(p[i]) > fre * f_insitu(cores-p[i]):
            t_total[i] = f_sim(p[i]) + f_comm(p[i], cores-p[i])
        else:
            t_total[i] = fre * f_insitu(cores-p[i]) + f_comm(p[i], cores-p[i])                                       
    return np.min(t_total), p[np.argmin(t_total)]

def interactive_insitu(total_core,app_core,freq,method): 
    fre = float(1.0/freq)
    if method == "Synchronous":
        interactive_sync(f_nek, f_img, total_core, fre)
    elif method == "Asynchronous":
        interactive_async(f_nek, f_img, f_adios, total_core, app_core, fre)
    elif method == "Auto":
        interactive_auto(f_nek, f_img, f_adios, total_core, fre)

def f_nek(p):
    return (0.064+2711.049/(p))

def f_img(p):
    return (3.806 + 8.248/np.sqrt(p))

def f_adios(p1, p2):
    return p1 * p2 - p1 * p2 + 1.0