import os
import json
import fire
import matplotlib.pyplot as plt


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def stat_print(case): # PhiNet stats print
    pj = os.path.join
    fig = plt.figure(figsize=(30, 90))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.grid(True)

    MM = 1.01
    mm = 0.99
    sm = .95
    stats = json.load(open(pj('logs', case, 'stats.json')))
    #print(stats[0].keys())
    plt_dict = {loss: [] for loss in stats[0].keys() if "loss" in loss}
    for stat in stats:
        gs = stat["global_step"]
        #losses = dict()
        for loss in stat.keys():
            if "loss" in loss:
                #print(loss, plt_dict[loss])
                #if plt_dict[loss] is not None:
                plt_dict[loss].append((gs, stat[loss]))
                #else:
                #    plt_dict[loss] = [(gs, stat[loss])]

    #gs, loss = list(zip(*[(stat['global_step'], stat['train_*/loss_*']) for stat in stats]))
    #print(plt_dict)
    for n, (loss_name, gs_val) in enumerate(plt_dict.items()):
        ax = fig.add_subplot(len(plt_dict), 1, n+1)
        ax.grid(True)
        gs, loss = list(zip(*gs_val))
        ax.plot(gs, loss, 'tab:blue', linewidth=1., zorder=-1, alpha=.2)
        ax.plot(gs, smooth(loss, sm), 'tab:blue', linewidth=1., zorder=-1)
        M = max(loss)
        M *= MM
        m = min(loss)
        m *= (mm if m>0 else MM)
        #M = max(smooth(loss, sm))*MM
        #m = min(smooth(loss, sm))*mm
        ax.legend()
        ax.title.set_text(loss_name.split('/')[-1])
        #ax.set_ylabel(loss_name.split('/')[-1] + f'\n Smoothing {sm}')
        ax.set_ylabel(f'Smoothing {sm}')
        ax.set_xlabel('step')
        #plt.xlabel('step')
        #plt.ylabel(loss_name.split('/')[-1] + f'\n Smoothing {sm}')
        #plt.xlabel('[step] - ' + loss_name.split('/')[-1] + f' / Smoothing {sm}')
        plt.ylim([m, M])

    fig.subplots_adjust(hspace=0.3)
    fig.savefig(pj('logs', case, f'loss_plot_{min(gs)}~{max(gs)}_smooth_{sm}.png'), transparent=True)

    #ax.plot(gs, loss, 'tab:blue', linewidth=1., zorder=-1, alpha=.2)
    #ax.plot(gs, smooth(loss, sm), 'tab:blue', linewidth=1., zorder=-1, label='')
    #M = max(smooth(loss, sm))*MM
    #m = min(smooth(loss, sm))*mm

    #stats = json.load(open(pj('ckpt', 'PhiNet-log_hilbert', 'PhiNet', 'stats.json')))
    #gs, loss = list(zip(*[(stat['global_step'], stat['Board/wav']) for stat in stats]))
    #ax.plot(gs, loss, 'tab:orange', linewidth=1., zorder=-1, alpha=.2)
    #ax.plot(gs, smooth(loss, sm), 'tab:orange', linewidth=1., zorder=-1, label='sparse phase')
    #M = max(M, max(smooth(loss, sm))*MM)
    #m = min(m, min(smooth(loss, sm))*mm)

    #ax.legend()
    #plt.xlabel('Steps')
    #plt.ylabel(f'L1 Error of Waveforms\n Smoothing {sm}')
    #plt.ylim([m, M])


    #fig.savefig(pj(src_dir, 'PhiNet stats.png'), transparent=True)

if __name__ == '__main__': fire.Fire(stat_print)

