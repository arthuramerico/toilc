import os
from toilc import solve

if not os.path.exists("graphs/"):
    os.makedirs("graphs/")

print("\n===============First Experiment===============")
print("\nResults when Alice has perfect knowledge about X:")
solve(pdfOutput="graphs/Experiment1_PerfectKnowledge.pdf",noPlot=True)
print("\nResults when Alice is completely ignorant about X:")
solve(pdfOutput="graphs/Experiment1_NoKnowledge.pdf", noise=1, noPlot=True)

print("\n===============Second Experiment===============")
print("\nResults for Figure 6:")
solve(pdfOutput="graphs/Experiment2_Figure6.pdf", distX="experimentData.txt", eeps=2, delta=0, noPlot=True)


print("\n===============Third Experiment===============")
print("\nResults for Figure 7:")
solve(pdfOutput="graphs/Experiment3_Figure7.pdf", distX="experimentData.txt", eeps=2, delta=0, noise=1, noPlot=True)

print("\nResults for Figure 8:")
solve(pdfOutput="graphs/Experiment3_Figure8.pdf", distX="experimentData.txt", eeps=2, delta=0.15, deltaLeft=0.15, tailMult=1.5, noise=1, noPlot=True)

print("\nThe charts obtained from the experiments can be found in the 'graphs' folder.")

