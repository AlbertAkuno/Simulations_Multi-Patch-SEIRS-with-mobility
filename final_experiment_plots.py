import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_data = {}
for n in [103, 203, 303, 403, 503]:
    random_data[n] = np.array(json.load(open(f'/Users/albertakuno/Desktop/forward_simulations/final_experiment/random_forward_simulation_n_{n}.json', 'r'))["sol"])

print(random_data[103].shape)

estimated_data = np.array(json.load(open('/Users/albertakuno/Desktop/forward_simulations/final_experiment/forward_simulation_n_503.json', 'r'))["sol"])
print(estimated_data.shape)

single_data = np.array(json.load(open('/Users/albertakuno/Desktop/forward_simulations/final_experiment/single_forward_simulation_n_503.json', 'r'))["sol"])
print(single_data.shape)

##########################################################################################
df =  pd.read_csv("./bbmm-drive/Casos_Diarios_Municipio_Confirmados_20211105.csv")
hermosillo_data = df.loc[df["cve_ent"]==26030]

time_series = hermosillo_data.squeeze()
time_series.head()

del time_series['cve_ent']
del time_series['poblacion']
del time_series['nombre']

time_series.tail(10)

time_series.index = pd.to_datetime(time_series.index, dayfirst=True, format="%d-%m-%Y")
print(time_series.index.dtype)

df_period = time_series.truncate(before="2021-09-21", after="2021-10-03")
df_period.head()
############################################################################################

linestyle = ['k-', 'k-.', 'k-o', 'k-+', 'k-^']
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
for i, n in enumerate([103, 203, 303, 403, 503]):
    sol_multipatch_sum = {
        "S": random_data[n][:, 0, :, 0].sum(axis=1),
        "E": random_data[n][:, 1, :, 0].sum(axis=1),
        "I": random_data[n][:, 2, :, 0].sum(axis=1),
        "R": random_data[n][:, 3, :, 0].sum(axis=1),
    }
    ax[0][0].plot(sol_multipatch_sum["S"][0:50], linestyle[i], label=f"$n$={n}", markersize=3, linewidth=1)
    ax[0][0].set_title("Susceptibles", fontsize=25)
    ax[0][1].plot(sol_multipatch_sum["E"][0:50], linestyle[i], label=f"$n$={n}", markersize=3, linewidth=1)
    ax[0][1].set_title("Exposed", fontsize=25)
    ax[1][0].plot(sol_multipatch_sum["I"][0:50], linestyle[i], label=f"$n$={n}", markersize=3, linewidth=1)
    ax[1][0].set_title("Infected", fontsize=25)
    ax[1][1].plot(sol_multipatch_sum["R"][0:50], linestyle[i], label=f"$n$={n}", markersize=3, linewidth=1)
    ax[1][1].set_title("Recovered", fontsize=25)

sol_single_dict = {
    "S": single_data[0, :],
    "E": single_data[1, :],
    "I": single_data[2, :],
    "R": single_data[3, :],
}

ax[0][0].plot(estimated_data[:, 0, :, 0].sum(axis=1)[0:50], 'k--', label="Estimated")
ax[0][1].plot(estimated_data[:, 1, :, 0].sum(axis=1)[0:50], 'k--', label="Estimated")
ax[1][0].plot(estimated_data[:, 2, :, 0].sum(axis=1)[0:50], 'k--', label="Estimated")
ax[1][1].plot(estimated_data[:, 3, :, 0].sum(axis=1)[0:50], 'k--', label="Estimated")
#ax[1][0].plot(range(len(df_period)), df_period, 'ro', markersize=4, label="Actual data points")

#ax[0][0].plot(sol_single_dict["S"], 'r--', linewidth=0.5, label="Single SEIR")
#ax[0][1].plot(sol_single_dict["E"], 'r--', linewidth=0.5, label="Single SEIR")
#ax[1][0].plot(sol_single_dict["I"], 'r--', linewidth=0.5, label="Single SEIR")
#ax[1][1].plot(sol_single_dict["R"], 'r--', linewidth=0.5, label="Single SEIR")

ax[0][0].set_ylabel(r"count", fontsize=20)
ax[0][0].set_xlabel(r"time", fontsize=20)
ax[0][1].set_ylabel(r"count", fontsize=20)
ax[0][1].set_xlabel(r"time", fontsize=20)
ax[1][0].set_xlabel(r"time", fontsize=20)
ax[1][0].set_ylabel(r"count", fontsize=20)
ax[1][1].set_xlabel(r"time", fontsize=20)
ax[1][1].set_ylabel(r"count", fontsize=20)

ax[0][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[1][1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax[0][0].grid()
#ax[0][0].legend()
ax[0][1].grid()
#ax[0][1].legend()
ax[1][0].grid()
#ax[1][0].legend()
ax[1][1].grid()
#ax[1][1].legend()
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.78, 1.0, 0, -0.4), fontsize=20, loc=4, ncol=3, bbox_transform=fig.transFigure)
plt.savefig("/Users/albertakuno/Desktop/figures/comparison_single_multi_t_50.jpg", format = "jpg", dpi=300, bbox_inches='tight')

# = plt.plot(hermosillo_data.values[3:])
#plt.show()


