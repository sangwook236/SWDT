#!/usr/bin/env python

#%matplotlib notebook

import seaborn as sns
import matplotlib.pyplot as plt

# REF [site] >> https://seaborn.pydata.org/tutorial/introduction.html
def introduction():
	# Apply the default theme.
	sns.set_theme()

	# Load an example dataset.
	tips = sns.load_dataset("tips")

	if False:
		# Create a visualization.
		sns.relplot(
			data=tips,
			x="total_bill", y="tip", col="time",
			hue="smoker", style="smoker", size="size",
		)

	#-----
	# A high-level API for statistical graphics.

	if True:
		dots = sns.load_dataset("dots")
		sns.relplot(
			data=dots, kind="line",
			x="time", y="firing_rate", col="align",
			hue="choice", size="coherence", style="choice",
			facet_kws=dict(sharex=False),
		)

		# Statistical estimation.
		fmri = sns.load_dataset("fmri")
		sns.relplot(
			data=fmri, kind="line",
			x="timepoint", y="signal", col="region",
			hue="event", style="event",
		)

		# Linear regression model.
		sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")

		# Distributional representations.
		sns.displot(data=tips, x="total_bill", col="time", kde=True)  # Histogram &  kernel density estimation.
		sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)  # Empirical cumulative distribution function.

		# Plots for categorical data.
		sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker")
		sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker", split=True)  # Kernel density estimation.
		sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")  # Mean value & confidence interval.

		# Multivariate views on complex datasets.
		penguins = sns.load_dataset("penguins")
		sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")  # Single relationship.
		sns.pairplot(data=penguins, hue="species")  # Joint and marginal distributions for all pairwise relationships and for each variable, respectively.

		# Lower-level tools for building figures.
		g = sns.PairGrid(penguins, hue="species", corner=True)
		g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
		g.map_lower(sns.scatterplot, marker="+")
		g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
		g.add_legend(frameon=True)
		g.legend.set_bbox_to_anchor((.61, .6))

	#-----
	# Opinionated defaults and flexible customization.

	if False:
		penguins = sns.load_dataset("penguins")

		sns.relplot(
			data=penguins,
			x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g"
		)

		sns.set_theme(style="ticks", font_scale=1.25)
		g = sns.relplot(
			data=penguins,
			x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g",
			palette="crest", marker="x", s=100,
		)
		g.set_axis_labels("Bill length (mm)", "Bill depth (mm)", labelpad=10)
		g.legend.set_title("Body mass (g)")
		g.figure.set_size_inches(6.5, 4.5)
		g.ax.margins(.15)
		g.despine(trim=True)

	plt.show()

def main():
	print(f"Version = {sns.__version__}.")

	introduction()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
