import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# 设置全局字体大小
fontsize = 20
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = fontsize
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['mathtext.default'] = 'regular'
# eps = 1e-5, 1e-6, 8e-6
# 1e-5
hci_1 = [-113.1147796351, -115.5826146086, -116.0575594459, -116.3417615981, -116.4887712540,-116.5450409219,
         -116.5972413643, -116.6017021615, -116.6017021615, -116.6019928620, -116.6020456267, -116.6020640326]
det_1 = [5510,             68090,           117753,          492446,          1688882,        3188686,
         4046520,          4552703,         4552703,         4759851,         4815450,        4836575]
det_1 = np.array(det_1)/1e6
# 1e-6
#hci_2 = [-111.8017784409, -115.4178598396, -116.1069611271, -116.4010398394]
#det_2 = [6462,             319880,          885583,          7121083]
# 8e-6
hci_3 = [-113.1187574601, -115.5878354122, -116.0790829724, -116.3584389438, -116.4947053362, -116.5573606185]
det_3 = [5838,             85423,           150992,          635344,          2203721,         4345607]
det_3 = np.array(det_3)/1e6

newqk = [-113.16636073540117, -116.39919624066263, -116.50324768206777, -116.54423947471705, -116.57950315043634,
         -116.58093852698008, -116.58206714568098, -116.58307068369125, -116.58834182998964, -116.58907211646148,
         -116.5898189555969,  -116.59046056421734, -116.5911301777943,  -116.59189107813428, -116.60063631039397,
         -116.60147220522693, -116.60149071979689, -116.60189217005488, -116.60201433131692, -116.60207114532258,
         -116.60208164329985, -116.60213993158443, -116.60216020187144, -116.60217097209794, -116.60241390638735,
         -116.60252936867137, -116.60256943521034]
det_newqk = [7876,   56024,   83416, 171728, 428712,
             450723, 470752, 488314, 617179, 640949,
             662792, 683016, 703017, 727037, 2194979,
             3161682, 3193606, 4060054, 4349852, 4522172,
             4554367, 4741041, 4809280, 4844813, 5766814, 7175474, 7479246]
det_newqk = np.array(det_newqk)/1e6

dmrg = [-116.6056091 for i in range(len(det_newqk))]

# 设置图形
plt.figure(figsize=(12, 8))
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.plot(det_1[3:], hci_1[3:], color="b", linewidth=2.5, linestyle="-.")
plt.scatter(det_1[3:], hci_1[3:], color="b", marker='v', s=200, label=r"$\epsilon_1 = 1e-5$")
plt.plot(det_3[3:], hci_3[3:], color="r", linewidth=2.5, linestyle="-.")
plt.scatter(det_3[3:], hci_3[3:], color="r", marker='o', s=200, label=r"$\epsilon_1 = 8e-6$")
plt.plot(det_newqk[2:-3], newqk[2:-3], color="m", linewidth=2.5, linestyle="-")
plt.scatter(det_newqk[2:-3], newqk[2:-3], color="m", marker='d', s=200, label="Qiankunnet-SCI")
plt.plot(det_newqk[2:-3], dmrg[2:-3], color="gray", linewidth=2.5, linestyle="-", label="DMRG")

plt.legend(loc="upper right", fontsize=fontsize)
plt.xlabel("Subspace dimension[Millions]", fontsize=fontsize)
plt.ylabel("Energy (Ha)", fontsize=fontsize)
#plt.xlabel("Bond Length (Bohr)", fontsize=fontsize)
plt.title(r"Fe$_2$S$_2$ Qiankunnet-SCI VS HCI", fontsize=fontsize)
# 调整子图布局
plt.tight_layout()
plt.show()
#DIR = "/Users/kanbowen/Desktop/jctc图片/new1"
#save_name = DIR + "/Fe2S2.pdf"

#plt.savefig(save_name, dpi=150, bbox_inches='tight')
#print(f"save as file: {save_name}")