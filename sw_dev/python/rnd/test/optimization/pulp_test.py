#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/coin-or/pulp
#	https://coin-or.github.io/pulp/

import pulp

# REF [site] >> https://github.com/coin-or/pulp
def simple_tutorial():
	# Create new variables.
	x = pulp.LpVariable("x", 0, 3)
	y = pulp.LpVariable("y", 0, 1)

	# Create a new problem.
	prob = pulp.LpProblem("myProblem", pulp.LpMinimize)

	# Combine variables to create expressions and constraints.
	prob += -4 * x + y

	prob += x + y <= 2

	# Solve the  problem.
	status = prob.solve()
	#status = prob.solve(solver=pulp.GLPK(msg=0))  # sudo apt install glpk-utils

	# Display the status of the solution.
	print("Status: {}.".format(pulp.LpStatus[status]))

	# Get the value of the variables.
	print("x = {}.".format(pulp.value(x)))
	print("y = {}.".format(pulp.value(y)))

# REF [site] >> https://coin-or.github.io/pulp/main/amply.html
#	Amply allows you to load and manipulate AMPL data as Python data structures.
def amply_quickstart_guide():
	from pulp import Amply

	# A simple set. Sets behave a lot like lists.
	data = Amply("set CITIES := Auckland Wellington Christchurch;")
	print("Cities: {}.".format(data.CITIES))
	print("Cities: {}.".format(data['CITIES']))
	assert data.CITIES == ['Auckland', 'Hamilton', 'Wellington']
	for c in data.CITIES:
		print(c)

	# Data can be integers, reals, symbolic, or quoted strings.
	data = Amply("""
	set BitsNPieces := 0 3.2 -6e4 Hello "Hello, World!";
	""")
	print("BitsNPieces: {}.".format(data.BitsNPieces))

	# Sets can contain multidimensional data.
	data = Amply("""
	set pairs dimen 2;
	set pairs := (1, 2) (2, 3) (3, 4);
	""")
	print("pairs: {}.".format(data.pairs))

	#--------------------
	#Amply.load_string(string)
	#Amply.load_file(file)  # Parse contents of file or file-like object.
	#Amply.load_string(file)  # Static. Create Amply object from contents of file or file-like object.

	data = Amply("""
	set CITIES;
	set ROUTES dimen 2;
	param COSTS{ROUTES};
	param DISTANCES{ROUTES};
	""")

	for data_file in ('cities.dat', 'routes.dat', 'costs.dat', 'distances.dat'):
		data.load_file(open(data_file))
		
	#--------------------
	# Sets themselves can be multidimensional (i.e. be subscriptable).

	# The set COUNTRIES didn’t actually have to exist itself.
	# Amply does not perform any validation on subscripts, it only uses them to figure out how many subscripts a set has.
	data = Amply("""
	set CITIES{COUNTRIES};
	set CITIES[Australia] := Adelaide Melbourne Sydney;
	set CITIES[Italy] := Florence Milan Rome;
	""")
	print("CITIES['Australia']: {}.".format(data.CITIES['Australia']))
	print("CITIES['Italy']: {}.".format(data.CITIES['Italy']))

	# To specify more than one, separate them by commas.
	data = Amply("""
	set SUBURBS{COUNTRIES, CITIES};
	set SUBURBS[Australia, Melbourne] := Docklands 'South Wharf' Kensington;
	""")
	print("SUBURBS['Australia', 'Melbourne']: {}.".format(data.SUBURBS['Australia', 'Melbourne']))

	#--------------------
	# Slices can be used to simplify the entry of multi-dimensional data.
	data = Amply("""
	set TRIPLES dimen 3;
	set TRIPLES := (1, 1, *) 2 3 4 (*, 2, *) 6 7 8 9 (*, *, *) (1, 1, 1);
	""")
	print("TRIPLES: {}.".format(data.TRIPLES))  # [(1, 1, 2), (1, 1, 3), (1, 1, 4), (6, 2, 7), (8, 2, 9), (1, 1, 1)].

	# Set data can also be specified using a matrix notation.

	# A '+' indicates that the pair is included in the set whereas a '-' indicates a pair not in the set.
	data = Amply("""
	set ROUTES dimen 2;
	set ROUTES : A B C D :=
			E + - - +
			F + + - -
	;
	""")
	print("ROUTES: {}.".format(data.ROUTES))  # [('E', 'A'), ('E', 'D'), ('F', 'A'), ('F', 'B')].

	# Matrices can also be transposed.
	data = Amply("""
	set ROUTES dimen 2;
	set ROUTES (tr) : E F :=
					A + +
					B - +
					C - -
					D + -
	;
	""")
	print("ROUTES: {}.".format(data.ROUTES))  # [('E', 'A'), ('F', 'A'), ('F', 'B'), ('E', 'D')].

	# Matrices only specify 2d data, however they can be combined with slices to define higher-dimensional data.
	data = Amply("""
	set QUADS dimen 2;
	set QUADS :=
	(1, 1, *, *) : 2 3 4 :=
				2 + - +
				3 - + +
	(1, 2, *, *) : 2 3 4 :=
				2 - + -
				3 + - -
	;
	""")
	print("QUADS: {}.".format(data.QUADS))  # [(1, 1, 2, 2), (1, 1, 2, 4), (1, 1, 3, 3), (1, 1, 3, 4), (1, 2, 2, 3), (1, 2, 3, 2)].

	#--------------------
	# Parameters are also supported.
	data = Amply("""
	param T := 30;
	param n := 5;
	""")
	print("T: {}.".format(data.T))
	print("n: {}.".format(data.n))

	# Parameters are commonly indexed over sets.
	# No validation is done by Amply, and the sets do not have to exist.
	# Parameter objects are represented as a mapping.
	data = Amply("""
	param COSTS{PRODUCTS};
	param COSTS :=
		FISH 8.5
		CARROTS 2.4
		POTATOES 1.6
	;
	""")
	print("COSTS: {}.".format(data.COSTS))
	print("COSTS['FISH']: {}.".format(data.COSTS['FISH']))

	# Parameter data statements can include a default clause.
	# If a '.' is included in the data, it is replaced with the default value.
	data = Amply("""
	param COSTS{P};
	param COSTS default 2 :=
	F 2
	E 1
	D .
	;
	""")
	print("COSTS['D']: {}.".format(data.COSTS['D']))

	# Parameter declarations can also have a default clause.
	# For these parameters, any attempt to access the parameter for a key that has not been defined will return the default value.
	data = Amply("""
	param COSTS{P} default 42;
	param COSTS :=
	F 2
	E 1
	;
	""")
	print("COSTS['DOES NOT EXIST']: {}.".format(data.COSTS['DOES NOT EXIST']))

	# Parameters can be indexed over multiple sets.
	# The resulting values can be accessed by treating the parameter object as a nested dictionary, or by using a tuple as an index.
	data = Amply("""
	param COSTS{CITIES, PRODUCTS};
	param COSTS :=
		Auckland FISH 5
		Auckland CHIPS 3
		Wellington FISH 4
		Wellington CHIPS 1
	;
	""")
	print("COSTS: {}.".format(data.COSTS))  # {'Wellington': {'FISH': 4.0, 'CHIPS': 1.0}, 'Auckland': {'FISH': 5.0, 'CHIPS': 3.0}}.
	print("COSTSCOSTS['Wellington']['CHIPS']: {}.".format(data.COSTSCOSTS['Wellington']['CHIPS']))  # Nested dict as key.
	print("COSTS['Wellington', 'CHIPS']: {}.".format(data.COSTS['Wellington', 'CHIPS']))  # Tuple as key.

	# Parameters support a slice syntax similar to that of sets.
	data = Amply("""
	param COSTS{CITIES, PRODUCTS};
	param COSTS :=
		[Auckland, *]
			FISH 5
			CHIPS 3
		[Wellington, *]
			FISH 4
			CHIPS 1
	;
	""")
	print("COSTS: {}.".format(data.COSTS))  # {'Wellington': {'FISH': 4.0, 'CHIPS': 1.0}, 'Auckland': {'FISH': 5.0, 'CHIPS': 3.0}}.

	# Parameters indexed over two sets can also be specified in tabular format.
	data = Amply("""
	param COSTS{CITIES, PRODUCTS};
	param COSTS: FISH CHIPS :=
		Auckland    5    3
		Wellington  4    1
	;
	""")
	print("COSTS: {}.".format(data.COSTS))  # {'Wellington': {'FISH': 4.0, 'CHIPS': 1.0}, 'Auckland': {'FISH': 5.0, 'CHIPS': 3.0}}.

	# Tabular data can also be transposed.
	data = Amply("""
	param COSTS{CITIES, PRODUCTS};
	param COSTS (tr): Auckland Wellington :=
			FISH   5        4
			CHIPS  3        1
	;
	""")
	print("COSTS: {}.".format(data.COSTS))  # {'Wellington': {'FISH': 4.0, 'CHIPS': 1.0}, 'Auckland': {'FISH': 5.0, 'CHIPS': 3.0}}.

	# Slices can be combined with tabular data for parameters indexed over more than 2 sets.
	data = Amply("""
	param COSTS{CITIES, PRODUCTS, SIZE};
	param COSTS :=
	[Auckland, *, *] :   SMALL LARGE :=
					FISH  5     9
					CHIPS 3     5
	[Wellington, *, *] : SMALL LARGE :=
					FISH  4     7
					CHIPS 1     2
	;
	""")
	print("COSTS: {}.".format(data.COSTS))  # {'Wellington': {'FISH': {'SMALL': 4.0, 'LARGE': 7.0}, 'CHIPS': {'SMALL': 1.0, 'LARGE': 2.0}}, 'Auckland': {'FISH': {'SMALL': 5.0, 'LARGE': 9.0}, '.

# REF [site] >> https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html
def configure_solver_user_guide():
	print("Solver list: {}.".format(pulp.listSolvers()))
	print("Solver list (available): {}.".format(pulp.listSolvers(onlyAvailable=True)))

	solver = pulp.getSolver("CPLEX_CMD")
	solver = pulp.getSolver("CPLEX_CMD", timeLimit=10)

	path_to_cplex = r"C:\Program Files\IBM\ILOG\CPLEX_Studio128\cplex\bin\x64_win64\cplex.exe"
	solver = pulp.CPLEX_CMD(path=path_to_cplex)

# REF [site] >> https://coin-or.github.io/pulp/guides/how_to_export_models.html
def import_and_export_models_user_guide():
	prob = pulp.LpProblem("test_export_dict_MIP", pulp.LpMinimize)
	x = pulp.LpVariable("x", 0, 4)
	y = pulp.LpVariable("y", -1, 1)
	z = pulp.LpVariable("z", 0, None, pulp.LpInteger)
	prob += x + 4 * y + 9 * z, "obj"
	prob += x + y <= 5, "c1"
	prob += x + z >= 10, "c2"
	prob += -y + z == 7.5, "c3"

	#--------------------
	# Export the problem into a dictionary.
	data = prob.to_dict()
	print("Problem: {}.".format(data))

	# Import the dictionary.
	var1, prob1 = pulp.LpProblem.from_dict(data)
	print("Loaded variable: {}.".format(var1))
	print("Loaded problem: {}.".format(prob1))

	prob1.solve()

	print("{} = {}.".format(var1['x'].name, var1['x'].value()))

	#--------------------
	# Export the problem into an mps file.
	prob.writeMPS("./test.mps")

	var2, prob2 = pulp.LpProblem.fromMPS("./test.mps")
	print("Loaded variable: {}.".format(var2))
	print("Loaded problem: {}.".format(prob2))

	prob2.solve()

	print("{} = {}.".format(var2['x'].name, var2['x'].value()))

	#--------------------
	# Export the problem to a json file.
	prob.to_json("./test.json")

	var3, prob3 = pulp.LpProblem.from_json("./test.json")
	print("Loaded variable: {}.".format(var3))
	print("Loaded problem: {}.".format(prob3))

	prob3.solve()

	print("{} = {}.".format(var3['x'].name, var3['x'].value()))

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html
def simplified_blending_problem():
	# Create the 'prob' variable to contain the problem data.
	prob = pulp.LpProblem("The Whiskas Problem", pulp.LpMinimize)

	pulp.LpVariable("example", None, 100)
	#pulp.LpVariable("example", upBound=100)

	# The 2 variables Beef and Chicken are created with a lower limit of zero.
	x1 = pulp.LpVariable("ChickenPercent", 0, None, pulp.LpInteger)
	x2 = pulp.LpVariable("BeefPercent", 0)

	# The objective function is added to 'prob' first.
	prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"

	# The five constraints are entered.
	prob += x1 + x2 == 100, "PercentagesSum"
	prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
	prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
	prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
	prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

	# The problem data is written to an .lp file.
	prob.writeLP("./WhiskasModel.lp")

	# The problem is solved using PuLP's choice of Solver.
	prob.solve()

	# The status of the solution is printed to the screen.
	print("Status: {}.".format(pulp.LpStatus[prob.status]))

	# Each of the variables is printed with it's resolved optimum value.
	for v in prob.variables():
		print("\t{} = {}.".format(v.name, v.varValue))

	# The optimised objective function value is printed to the screen
	print("Total Cost of Ingredients per can = {}.".format(pulp.value(prob.objective)))

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html
def full_blending_problem():
	# Creates a list of the Ingredients
	Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']

	# A dictionary of the costs of each of the Ingredients is created.
	costs = {
		'CHICKEN': 0.013, 
		'BEEF': 0.008, 
		'MUTTON': 0.010, 
		'RICE': 0.002, 
		'WHEAT': 0.005, 
		'GEL': 0.001
	}

	# A dictionary of the protein percent in each of the Ingredients is created.
	proteinPercent = {
		'CHICKEN': 0.100, 
		'BEEF': 0.200, 
		'MUTTON': 0.150, 
		'RICE': 0.000, 
		'WHEAT': 0.040, 
		'GEL': 0.000
	}
	# A dictionary of the fat percent in each of the Ingredients is created.
	fatPercent = {
		'CHICKEN': 0.080, 
		'BEEF': 0.100, 
		'MUTTON': 0.110, 
		'RICE': 0.010, 
		'WHEAT': 0.010, 
		'GEL': 0.000
	}
	# A dictionary of the fibre percent in each of the Ingredients is created.
	fibrePercent = {
		'CHICKEN': 0.001, 
		'BEEF': 0.005, 
		'MUTTON': 0.003, 
		'RICE': 0.100, 
		'WHEAT': 0.150, 
		'GEL': 0.000
	}
	# A dictionary of the salt percent in each of the Ingredients is created.
	saltPercent = {
		'CHICKEN': 0.002, 
		'BEEF': 0.005, 
		'MUTTON': 0.007, 
		'RICE': 0.002, 
		'WHEAT': 0.008, 
		'GEL': 0.000
	}

	# Create the 'prob' variable to contain the problem data.
	prob = pulp.LpProblem("The Whiskas Problem", pulp.LpMinimize)

	# A dictionary called 'ingredient_vars' is created to contain the referenced variables.
	ingredient_vars = pulp.LpVariable.dicts("Ingr", Ingredients, 0)

	# The objective function is added to 'prob' first.
	prob += pulp.lpSum([costs[i] * ingredient_vars[i] for i in Ingredients]), "Total Cost of Ingredients per can"

	# The five constraints are added to 'prob'.
	prob += pulp.lpSum([ingredient_vars[i] for i in Ingredients]) == 100, "PercentagesSum"
	prob += pulp.lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0, "ProteinRequirement"
	prob += pulp.lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0, "FatRequirement"
	prob += pulp.lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0, "FibreRequirement"
	prob += pulp.lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4, "SaltRequirement"

	# The problem data is written to an .lp file.
	prob.writeLP("./WhiskasModel2.lp")

	# The problem is solved using PuLP's choice of Solver.
	prob.solve()

	# The status of the solution is printed to the screen.
	print("Status: {}.".format(pulp.LpStatus[prob.status]))

	# Each of the variables is printed with it's resolved optimum value.
	for v in prob.variables():
		print("\t{} = {}.".format(v.name, v.varValue))

	# The optimised objective function value is printed to the screen
	print("Total Cost of Ingredients per can = {}.".format(pulp.value(prob.objective)))

def set_partitioning_problem():
	max_tables = 5
	max_table_size = 4
	guests = "A B C D E F G I J K L M N O P Q R".split()

	def happiness(table):
		"""
		Find the happiness of the table
		- by calculating the maximum distance between the letters.
		"""
		return abs(ord(table[0]) - ord(table[-1]))

	# Create list of all possible tables.
	possible_tables = [tuple(c) for c in pulp.allcombinations(guests, max_table_size)]

	# Create a binary variable to state that a table setting is used.
	x = pulp.LpVariable.dicts("table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger)

	seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)

	seating_model += pulp.lpSum([happiness(table) * x[table] for table in possible_tables])

	# Specify the maximum number of tables.
	seating_model += pulp.lpSum([x[table] for table in possible_tables]) <= max_tables, "Maximum_number_of_tables"

	# A guest must seated at one and only one table.
	for guest in guests:
		seating_model += pulp.lpSum([x[table] for table in possible_tables if guest in table]) == 1, "Must_seat_{}".format(guest)

	if False:
		# I've taken the optimal solution from a previous solving. x is the variable dictionary.
		solution = {
			("M", "N"): 1.0,
			("E", "F", "G"): 1.0,
			("A", "B", "C", "D"): 1.0,
			("I", "J", "K", "L"): 1.0,
			("O", "P", "Q", "R"): 1.0,
		}
		for k, v in solution.items():
			x[k].setInitialValue(v)
			#x[k].fixValue()

	if True:
		seating_model.solve()
	else:
		# I usually turn msg=True so I can see the messages from the solver confirming it loaded the solution correctly.
		solver = pulp.PULP_CBC_CMD(msg=True, warmStart=True)
		#solver = pulp.CPLEX_CMD(msg=True, warmStart=True)
		#solver = pulp.GUROBI_CMD(msg=True, warmStart=True)
		#solver = pulp.CPLEX_PY(msg=True, warmStart=True)
		#solver = pulp.GUROBI(msg=True, warmStart=True)
		seating_model.solve(solver)

	print("The choosen tables are out of a total of {}:".format(len(possible_tables)))
	for table in possible_tables:
		if x[table].value() == 1.0:
			print(table)

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
def sudoku_problem_1():
	# A list of strings from "1" to "9" is created.
	Sequence = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

	# The Vals, Rows and Cols sequences all follow this form.
	Vals = Sequence
	Rows = Sequence
	Cols = Sequence

	# The boxes list is created, with the row and column index of each square in each box.
	Boxes =[]
	for i in range(3):
		for j in range(3):
			Boxes += [[(Rows[3*i+k], Cols[3*j+l]) for k in range(3) for l in range(3)]]

	# The prob variable is created to contain the problem data.
	prob = pulp.LpProblem("Sudoku Problem", pulp.LpMinimize)

	# The problem variables are created.
	choices = pulp.LpVariable.dicts("Choice", (Vals, Rows, Cols), 0, 1, pulp.LpInteger)

	# The arbitrary objective function is added.
	prob += 0, "Arbitrary Objective Function"

	# A constraint ensuring that only one value can be in each square is created.
	for r in Rows:
		for c in Cols:
			prob += pulp.lpSum([choices[v][r][c] for v in Vals]) == 1, ""

	# The row, column and box constraints are added for each value.
	for v in Vals:
		for r in Rows:
			prob += pulp.lpSum([choices[v][r][c] for c in Cols]) == 1, ""
			
		for c in Cols:
			prob += pulp.lpSum([choices[v][r][c] for r in Rows]) == 1, ""

		for b in Boxes:
			prob += pulp.lpSum([choices[v][r][c] for (r, c) in b]) == 1, ""

	# The starting numbers are entered as constraints.
	prob += choices["5"]["1"]["1"] == 1, ""
	prob += choices["6"]["2"]["1"] == 1, ""
	prob += choices["8"]["4"]["1"] == 1, ""
	prob += choices["4"]["5"]["1"] == 1, ""
	prob += choices["7"]["6"]["1"] == 1, ""
	prob += choices["3"]["1"]["2"] == 1, ""
	prob += choices["9"]["3"]["2"] == 1, ""
	prob += choices["6"]["7"]["2"] == 1, ""
	prob += choices["8"]["3"]["3"] == 1, ""
	prob += choices["1"]["2"]["4"] == 1, ""
	prob += choices["8"]["5"]["4"] == 1, ""
	prob += choices["4"]["8"]["4"] == 1, ""
	prob += choices["7"]["1"]["5"] == 1, ""
	prob += choices["9"]["2"]["5"] == 1, ""
	prob += choices["6"]["4"]["5"] == 1, ""
	prob += choices["2"]["6"]["5"] == 1, ""
	prob += choices["1"]["8"]["5"] == 1, ""
	prob += choices["8"]["9"]["5"] == 1, ""
	prob += choices["5"]["2"]["6"] == 1, ""
	prob += choices["3"]["5"]["6"] == 1, ""
	prob += choices["9"]["8"]["6"] == 1, ""
	prob += choices["2"]["7"]["7"] == 1, ""
	prob += choices["6"]["3"]["8"] == 1, ""
	prob += choices["8"]["7"]["8"] == 1, ""
	prob += choices["7"]["9"]["8"] == 1, ""
	prob += choices["3"]["4"]["9"] == 1, ""
	prob += choices["1"]["5"]["9"] == 1, ""
	prob += choices["6"]["6"]["9"] == 1, ""
	prob += choices["5"]["8"]["9"] == 1, ""

	# The problem data is written to an .lp file.
	prob.writeLP("./Sudoku.lp")

	# The problem is solved using PuLP's choice of Solver.
	prob.solve()

	# The status of the solution is printed to the screen.
	print("Status: {}.".format(pulp.LpStatus[prob.status]))

	# A file called sudokuout.txt is created/overwritten for writing to.
	sudokuout = open('./sudokuout.txt', 'w')

	# The solution is written to the sudokuout.txt file.
	for r in Rows:
		if r == "1" or r == "4" or r == "7":
			sudokuout.write("+-------+-------+-------+\n")
		for c in Cols:
			for v in Vals:
				if pulp.value(choices[v][r][c])==1:
					if c == "1" or c == "4" or c == "7":
						sudokuout.write("| ")

					sudokuout.write(v + " ")

					if c == "9":
						sudokuout.write("|\n")
	sudokuout.write("+-------+-------+-------+")
	sudokuout.close()

	# The location of the solution is give to the user.
	print("Solution Written to sudokuout.txt.")

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
def sudoku_problem_2():
	# All rows, columns and values within a Sudoku take values from 1 to 9.
	VALS = ROWS = COLS = range(1, 10)

	# The boxes list is created, with the row and column index of each square in each box.
	Boxes = [
		[(3 * i + k + 1, 3 * j + l + 1) for k in range(3) for l in range(3)]
		for i in range(3)
		for j in range(3)
	]

	# The prob variable is created to contain the problem data.
	prob = pulp.LpProblem("Sudoku Problem")

	# The decision variables are created.
	choices = pulp.LpVariable.dicts("Choice", (VALS, ROWS, COLS), cat="Binary")

	# We do not define an objective function since none is needed.

	# A constraint ensuring that only one value can be in each square is created.
	for r in ROWS:
		for c in COLS:
			prob += pulp.lpSum([choices[v][r][c] for v in VALS]) == 1

	# The row, column and box constraints are added for each value.
	for v in VALS:
		for r in ROWS:
			prob += pulp.lpSum([choices[v][r][c] for c in COLS]) == 1

		for c in COLS:
			prob += pulp.lpSum([choices[v][r][c] for r in ROWS]) == 1

		for b in Boxes:
			prob += pulp.lpSum([choices[v][r][c] for (r, c) in b]) == 1

	# The starting numbers are entered as constraints.
	input_data = [
		(5, 1, 1),
		(6, 2, 1),
		(8, 4, 1),
		(4, 5, 1),
		(7, 6, 1),
		(3, 1, 2),
		(9, 3, 2),
		(6, 7, 2),
		(8, 3, 3),
		(1, 2, 4),
		(8, 5, 4),
		(4, 8, 4),
		(7, 1, 5),
		(9, 2, 5),
		(6, 4, 5),
		(2, 6, 5),
		(1, 8, 5),
		(8, 9, 5),
		(5, 2, 6),
		(3, 5, 6),
		(9, 8, 6),
		(2, 7, 7),
		(6, 3, 8),
		(8, 7, 8),
		(7, 9, 8),
		(3, 4, 9),
		# Since the previous Sudoku contains only one unique solution, we remove some numers from the board to obtain a Sudoku with multiple solutions.
		#(1, 5, 9),
		#(6, 6, 9),
		#(5, 8, 9)
	]

	for (v, r, c) in input_data:
		prob += choices[v][r][c] == 1

	# The problem data is written to an .lp file.
	prob.writeLP("./Sudoku2.lp")

	# A file called sudokuout.txt is created/overwritten for writing to.
	sudokuout = open("./sudokuout2.txt", "w")

	while True:
		prob.solve()
		# The status of the solution is printed to the screen.
		print("Status: {}.".format(pulp.LpStatus[prob.status]))
		# The solution is printed if it was deemed "optimal" i.e met the constraints.
		if pulp.LpStatus[prob.status] == "Optimal":
			# The solution is written to the sudokuout.txt file.
			for r in ROWS:
				if r in [1, 4, 7]:
					sudokuout.write("+-------+-------+-------+\n")
				for c in COLS:
					for v in VALS:
						if pulp.value(choices[v][r][c]) == 1:
							if c in [1, 4, 7]:
								sudokuout.write("| ")
							sudokuout.write(str(v) + " ")
							if c == 9:
								sudokuout.write("|\n")
			sudokuout.write("+-------+-------+-------+\n\n")
			# The constraint is added that the same solution cannot be returned again.
			prob += (
				pulp.lpSum([
					choices[v][r][c]
					for v in VALS
					for r in ROWS
					for c in COLS
					if pulp.value(choices[v][r][c]) == 1
				])
				<= 80
			)
		# If a new optimal solution cannot be found, we end the program.
		else:
			break
	sudokuout.close()

	# The location of the solutions is give to the user.
	print("Solutions Written to sudokuout2.txt.")

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html
def transportation_problem():
	# Creates a list of all the supply nodes.
	Warehouses = ["A", "B"]
	# Creates a dictionary for the number of units of supply for each supply node.
	supply = {"A": 1000, "B": 4000}
	# Creates a list of all demand nodes.
	Bars = ["1", "2", "3", "4", "5"]
	# Creates a dictionary for the number of units of demand for each demand node.
	demand = {
		"1":500,
		"2":900,
		"3":1800,
		"4":200,
		"5":700,
	}

	# Creates a list of costs of each transportation path.
	costs = [
		# Bars.
		#1  2  3  4  5.
		[2, 4, 5, 2, 1], #A  Warehouses.
		[3, 1, 3, 2, 3]  #B
	]

	# The cost data is made into a dictionary.
	costs = pulp.makeDict([Warehouses, Bars], costs, 0)

	# Creates the 'prob' variable to contain the problem data.
	prob = pulp.LpProblem("Beer Distribution Problem", pulp.LpMinimize)

	# Creates a list of tuples containing all the possible routes for transport.
	Routes = [(w, b) for w in Warehouses for b in Bars]

	# A dictionary called 'Vars' is created to contain the referenced variables(the routes).
	vars = pulp.LpVariable.dicts("Route", (Warehouses, Bars), 0, None, pulp.LpInteger)

	# The objective function is added to 'prob' first.
	prob += pulp.lpSum([vars[w][b] * costs[w][b] for (w, b) in Routes]), "Sum_of_Transporting_Costs"

	# The supply maximum constraints are added to prob for each supply node (warehouse).
	for w in Warehouses:
		prob += pulp.lpSum([vars[w][b] for b in Bars]) <= supply[w], "Sum_of_Products_out_of_Warehouse_{}".format(w)

	# The demand minimum constraints are added to prob for each demand node (bar).
	for b in Bars:
		prob += pulp.lpSum([vars[w][b] for w in Warehouses]) >= demand[b], "Sum_of_Products_into_Bar{}".format(b)
					
	# The problem data is written to an .lp file.
	prob.writeLP("./BeerDistributionProblem.lp")

	# The problem is solved using PuLP's choice of Solver.
	prob.solve()

	# The status of the solution is printed to the screen.
	print("Status: {}.".format(pulp.LpStatus[prob.status]))

	# Each of the variables is printed with it's resolved optimum value.
	for v in prob.variables():
		print("\t{} = {}.".format(v.name, v.varValue))

	# The optimised objective function value is printed to the screen.
	print("Total Cost of Transportation = {}.".format(pulp.value(prob.objective)))

# REF [site] >> https://coin-or.github.io/pulp/CaseStudies/a_two_stage_production_planning_problem.html
def two_stage_production_planning_problem():
	# Parameters.
	products = ["wrenches", "pliers"]
	price = [130, 100]
	steel = [1.5, 1]
	molding = [1, 1]
	assembly = [0.3, 0.5]
	capsteel = 27
	capmolding = 21
	LB = [0, 0]
	capacity_ub = [15, 16]
	steelprice = 58
	scenarios = [0, 1, 2, 3]
	pscenario = [0.25, 0.25, 0.25, 0.25]
	wrenchearnings = [160, 160, 90, 90]
	plierearnings = [100, 100, 100, 100]
	capassembly = [8, 10, 8, 10]

	production = [(j, i) for j in scenarios for i in products]
	pricescenario = [[wrenchearnings[j], plierearnings[j]] for j in scenarios]
	priceitems = [item for sublist in pricescenario for item in sublist]

	# Create dictionaries for the parameters.
	price_dict = dict(zip(production, priceitems))
	capacity_dict = dict(zip(products, capacity_ub * 4))
	steel_dict = dict(zip(products, steel))
	molding_dict = dict(zip(products, molding))
	assembly_dict = dict(zip(products, assembly))

	# Create variables and parameters as dictionaries.
	production_vars = pulp.LpVariable.dicts("production", (scenarios, products), lowBound=0, cat="Continuous")
	steelpurchase = pulp.LpVariable("steelpurchase", lowBound=0, cat="Continuous")

	# Create the 'gemstoneprob' variable to specify.
	gemstoneprob = pulp.LpProblem("The Gemstone Tool Problem", pulp.LpMaximize)

	# The objective function is added to 'gemstoneprob' first.
	gemstoneprob += (
		pulp.lpSum(
			[
				pscenario[j] * (price_dict[(j, i)] * production_vars[j][i])
				for (j, i) in production
			]
			- steelpurchase * steelprice
		),
		"Total cost",
	)

	for j in scenarios:
		gemstoneprob += pulp.lpSum([steel_dict[i] * production_vars[j][i] for i in products]) - steelpurchase <= 0, ("Steel capacity" + str(j))
		gemstoneprob += pulp.lpSum([molding_dict[i] * production_vars[j][i] for i in products]) <= capmolding, ("molding capacity" + str(j))
		gemstoneprob += pulp.lpSum([assembly_dict[i] * production_vars[j][i] for i in products]) <= capassembly[j], ("assembly capacity" + str(j))
		for i in products:
			gemstoneprob += production_vars[j][i] <= capacity_dict[i], ("capacity " + str(i) + str(j))

	# Print problem.
	print(gemstoneprob)

	# The problem data is written to an .lp file.
	gemstoneprob.writeLP("./gemstoneprob.lp")

	# The problem is solved using PuLP's choice of Solver.
	gemstoneprob.solve()
	# The status of the solution is printed to the screen.
	print("Status: {}.".format(pulp.LpStatus[gemstoneprob.status]))

	# OUTPUT

	# Each of the variables is printed with it's resolved optimum value.
	for v in gemstoneprob.variables():
		print("\t{} = {}.".format(v.name, v.varValue))
	production = [v.varValue for v in gemstoneprob.variables()]

	# The optimised objective function value is printed to the console.
	print("Total price = {}.".format(pulp.value(gemstoneprob.objective)))

def main():
	#simple_tutorial()
	#amply_quickstart_guide()

	#configure_solver_user_guide()
	#import_and_export_models_user_guide()

	#--------------------
	#simplified_blending_problem()
	#full_blending_problem()

	# Set partitioning problem: all items in one set S must be contained in one and only one partition.
	# Set packing problem: all items in one set S must be contained in zero or one partitions.
	# Set covering problem: all items in one set S must be contained in at least one partition.
	set_partitioning_problem()

	#sudoku_problem_1()
	#sudoku_problem_2()

	#transportation_problem()
	#two_stage_production_planning_problem()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
