import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r=json.load(open("results.json")); c=r["curve"]
d=[x["d"] for x in c]
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(d,[x["exact"] for x in c],"k-",label="Exact (classical diagonalisation)")
ax[0].plot(d,[x["vqe"] for x in c],"o",label="VQE (UCCSD, ideal simulator)")
ax[0].plot(d,[x["hf"] for x in c],"--",label="Hartree-Fock")
ax[0].set_xlabel("H-H distance (Å)");ax[0].set_ylabel("Energy (Hartree)");ax[0].legend();ax[0].set_title("H2 / STO-3G dissociation curve")
n=json.load(open("noise_results.json"))
labels=["Ideal\n(no noise)","8192 shots","+1% CX\nnoise","+3% CX\nnoise"]
vals=[ [x for x in c if x["d"]==0.735][0]["err_mHa"], n["shots_only"]["err_mHa"], n["noise_1pct"]["err_mHa"], n["noise_3pct"]["err_mHa"]]
ax[1].bar(labels,[abs(v) for v in vals]);ax[1].axhline(1.6,ls="--",color="gray");ax[1].text(0.4,2.2,"chemical accuracy 1.6 mHa",color="gray")
ax[1].set_yscale("log");ax[1].set_ylabel("|Error vs exact| (milli-Hartree)");ax[1].set_title("Error at 0.735 Å")
plt.suptitle("Educational benchmark: VQE on H2 (not related to any drug)");plt.tight_layout();plt.savefig("h2_vqe_chart.png",dpi=130)
print(vals)
