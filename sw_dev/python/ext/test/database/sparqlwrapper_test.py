#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import SPARQLWrapper

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def select_example():
	sparql = SPARQLWrapper.SPARQLWrapper("http://vocabs.ardc.edu.au/repository/api/sparql/csiro_international-chronostratigraphic-chart_geologic-time-scale-2020")
	sparql.setReturnFormat(SPARQLWrapper.JSON)

	# Gets the first 3 geological ages from a Geological Timescale database, via a SPARQL endpoint.
	sparql.setQuery("""
PREFIX gts: <http://resource.geosciml.org/ontology/timescale/gts#>

SELECT *
WHERE {
	?a a gts:Age .
}
ORDER BY ?a
LIMIT 3
"""
	)

	try:
		ret = sparql.queryAndConvert()

		for r in ret["results"]["bindings"]:
			print(r)
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def ask_example():
	sparql = SPARQLWrapper.SPARQLWrapper("http://dbpedia.org/sparql")
	sparql.setQuery("""
ASK WHERE {
	<http://dbpedia.org/resource/Asturias> rdfs:label "Asturias"@es
}
"""
	)
	sparql.setReturnFormat(SPARQLWrapper.XML)

	try:
		results = sparql.query().convert()
		print(results.toxml())
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def construct_example():
	sparql = SPARQLWrapper.SPARQLWrapper("http://dbpedia.org/sparql")

	sparql.setQuery("""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX sdo: <https://schema.org/>

CONSTRUCT {
	?lang a sdo:Language ;
	sdo:alternateName ?iso6391Code .
}
WHERE {
	?lang a dbo:Language ;
	dbo:iso6391Code ?iso6391Code .
	FILTER (STRLEN(?iso6391Code)=2)  # To filter out non-valid values.
}
LIMIT 3
"""
	)

	try:
		results = sparql.queryAndConvert()
		print(results.serialize())
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def describe_example():
	sparql = SPARQLWrapper.SPARQLWrapper("http://dbpedia.org/sparql")
	sparql.setQuery("DESCRIBE <http://dbpedia.org/resource/Asturias>")

	try:
		results = sparql.queryAndConvert()
		print(results.serialize(format="json-ld"))
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def update_example():
	sparql = SPARQLWrapper.SPARQLWrapper("https://example.org/sparql")
	sparql.setHTTPAuth(SPARQLWrapper.DIGEST)
	sparql.setCredentials("some-login", "some-password")
	sparql.setMethod(SPARQLWrapper.POST)

	sparql.setQuery("""
PREFIX dbp:  <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

WITH <http://example.graph>
DELETE {
	dbo:Asturias rdfs:label "Asturies"@ast
}
"""
	)

	try:
		results = sparql.query()
		print(results.response.read())
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def SPARQLWrapper2_example():
	sparql = SPARQLWrapper.SPARQLWrapper2("http://dbpedia.org/sparql")
	sparql.setQuery("""
PREFIX dbp:  <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?label
WHERE {
	dbp:Asturias rdfs:label ?label
}
LIMIT 3
"""
	)

	try:
		for result in sparql.query().bindings:
			print(f"{result['label'].lang}, {result['label'].value}")
	except Exception as ex:
		print(ex)

# REF [site] >> https://sparqlwrapper.readthedocs.io/en/latest/main.html
def partial_interpretation_of_results():
	sparql = SPARQLWrapper.SPARQLWrapper2("http://example.org/sparql")
	sparql.setQuery("""
SELECT ?subj ?prop
WHERE {
	?subj ?prop ?obj
}
"""
	)

	try:
		ret = sparql.query()
		print(ret.variables)  # This is an array consisting of "subj" and "prop".
		for binding in ret.bindings:
			# Each binding is a dictionary. Let us just print the results.
			print(f"{binding['subj'].value}, {binding['subj'].type}")
			print(f"{binding['prop'].value}, {binding['prop'].type}")
	except Exception as ex:
		print(ex)

	#-----
	sparql.setQuery("""
SELECT ?subj ?obj ?opt
WHERE {
	?subj <http://a.b.c> ?obj .
	OPTIONAL {
		?subj <http://d.e.f> ?opt
	}
}
"""
	)

	try:
		ret = sparql.query()
		print(ret.variables)  # This is an array consisting of "subj", "obj", "opt".
		if ("subj", "prop", "opt") in ret:
			# There is at least one binding covering the optional "opt", too.
			bindings = ret["subj", "obj", "opt"]
			# Bindings is an array of dictionaries with the full bindings.
			for b in bindings:
				subj = b["subj"].value
				o = b["obj"].value
				opt = b["opt"].value

				# Do something nice with subj, o, and opt.

		# Another way of accessing to values for a single variable: take all the bindings of the "subj", "obj", "opt".
		subjbind = ret.getValues("subj")  # An array of Value instances.
		objbind = ret.getValues("obj")  # An array of Value instances.
		optbind = ret.getValues("opt")  # An array of Value instances.
	except Exception as ex:
		print(ex)

def dbpedia_test():
	sparql = SPARQLWrapper.SPARQLWrapper("http://dbpedia.org/sparql")
	sparql.setReturnFormat(SPARQLWrapper.JSON)

	if True:
		sparql.setQuery("""
SELECT ?uri ?name ?page ?nick
WHERE {
	?uri a foaf:Person ;
	foaf:name ?name;
	foaf:page ?page;
	foaf:nick ?nick.
}
LIMIT 100
"""
		)
	elif False:
		sparql.setQuery("""
SELECT ?name ?birth ?role
WHERE{
	?x a foaf:Person ;
	dbpprop:fullname ?name;
	dbpprop:countryofbirth ?birth;
	dbpprop:role ?role.

	FILTER regex(?birth, "land$").
	FILTER regex(?birth, "^Eng").
	FILTER regex(?birth, "England").
} LIMIT 100
"""
		)

	try:
		ret = sparql.queryAndConvert()
		print(ret["results"]["bindings"])
	except Exception as ex:
		print(ex)

def dbpedia_ko_test():
	sparql = SPARQLWrapper.SPARQLWrapper("http://ko.dbpedia.org/sparql")
	sparql.setReturnFormat(SPARQLWrapper.JSON)

	if False:
		sparql.setQuery("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://ko.dbpedia.org/property/>

SELECT DISTINCT ?comment
WHERE {
	?s foaf:name ?name;
	rdfs:comment ?comment;
	dbp:occupation ?occupation.
	FILTER(REGEX(STR(?occupation), '정치'))
}
LIMIT 30
"""
		)
	elif False:
		sparql.setQuery("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://ko.dbpedia.org/property/>

SELECT ?comment, ?relative, ?parent
WHERE {
	?s foaf:name ?name;
	rdfs:comment ?comment.
	FILTER(STR(?name) = '하정우')
	OPTIONAL{?relative dbo:relative ?s.}
	OPTIONAL{?parent dbo:child ?s.}
}
LIMIT 30
"""
		)
	elif True:
		sparql.setQuery("""
select * where {
	?s <http://ko.dbpedia.org/property/장소> ?o
} LIMIT 100
"""
		)
	elif False:
		sparql.setQuery("""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://ko.dbpedia.org/property/>
PREFIX res: <http://ko.dbpedia.org/resource/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

select * where {
	?s rdf:type foaf:Person.
	?s <http://ko.dbpedia.org/property/국가> '대한민국'@ko.
}
"""
		)
	elif False:
		sparql.setQuery("""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://ko.dbpedia.org/property/>
PREFIX res: <http://ko.dbpedia.org/resource/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
select count(*) where {
	?s rdf:type foaf:Person.
	{?s dbp:출생일 ?Bdate.} UNION {?s dbp:사망일 ?Ddate.}
	?s dbo:abstract ?abstract.
	?s dbp:국적 ?nation.
}
"""
		)

	try:
		ret = sparql.queryAndConvert()
		print(ret["results"]["bindings"])
	except Exception as ex:
		print(ex)

def main():
	#select_example()
	#ask_example()
	#construct_example()
	#describe_example()
	#update_example()

	#SPARQLWrapper2_example()

	#partial_interpretation_of_results()

	#-----
	dbpedia_test()
	dbpedia_ko_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
