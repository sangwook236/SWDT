#include "iniparser.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <unistd.h>
#include <cstdio>
#include <iostream>


namespace {
namespace local {

void create_example_ini_file(void)
{
	FILE	*	ini ;

	ini = fopen("data/application_configuration/iniparser/example.ini", "w");
	fprintf(ini, "\n\
#\n\
# This is an example of ini file\n\
#\n\
\n\
[Pizza]\n\
\n\
Ham       = yes ;\n\
Mushrooms = TRUE ;\n\
Capres    = 0 ;\n\
Cheese    = NO ;\n\
\n\
\n\
[Wine]\n\
\n\
Grape     = Cabernet Sauvignon ;\n\
Year      = 1989 ;\n\
Country   = Spain ;\n\
Alcohol   = 12.5  ;\n\
\n\
#\n\
# end of file\n\
#\n");

	fclose(ini);
}

int parse_ini_file(char * ini_name)
{
	dictionary	*	ini ;

	/* Some temporary variables to hold query results */
	int				b ;
	int				i ;
	double			d ;
	char		*	s ;

	ini = iniparser_load(ini_name);
	if (ini==NULL) {
		fprintf(stderr, "cannot parse file [%s]", ini_name);
		return -1 ;
	}
	iniparser_dump(ini, stderr);

	/* Get pizza attributes */
	printf("Pizza:\n");

	b = iniparser_getboolean(ini, "pizza:ham", -1);
	printf("Ham:       [%d]\n", b);
	b = iniparser_getboolean(ini, "pizza:mushrooms", -1);
	printf("Mushrooms: [%d]\n", b);
	b = iniparser_getboolean(ini, "pizza:capres", -1);
	printf("Capres:    [%d]\n", b);
	b = iniparser_getboolean(ini, "pizza:cheese", -1);
	printf("Cheese:    [%d]\n", b);

	/* Get wine attributes */
	printf("Wine:\n");
	s = iniparser_getstr(ini, "wine:grape");
	if (s) {
		printf("grape:     [%s]\n", s);
	} else {
		printf("grape:     not found\n");
	}
	i = iniparser_getint(ini, "wine:year", -1);
	if (i>0) {
		printf("year:      [%d]\n", i);
	} else {
		printf("year:      not found\n");
	}
	s = iniparser_getstr(ini, "wine:country");
	if (s) {
		printf("country:   [%s]\n", s);
	} else {
		printf("country:   not found\n");
	}
	d = iniparser_getdouble(ini, "wine:alcohol", -1.0);
	if (d>0.0) {
		printf("alcohol:   [%g]\n", d);
	} else {
		printf("alcohol:   not found\n");
	}

	//
	{
		FILE *fp = fopen("data/application_configuration/iniparser/result1.ini", "w");
		iniparser_dump(ini, fp);
		fclose(fp);
	}

	int ret;

	ret = iniparser_set(ini, "Pizza:Cheese", "YES");

	ret = iniparser_set(ini, "Bread", NULL);  // caution !!!: it's necessary
	ret = iniparser_set(ini, "Bread:Cream", "yes");
	ret = iniparser_set(ini, "Bread:Raisin", "no");

	ret = iniparser_set(ini, "Cake/Chocolate", NULL);  // caution !!!: it's necessary
	ret = iniparser_set(ini, "Cake/Chocolate:Chocolate", "Y");
	ret = iniparser_set(ini, "Cake/Chocolate:Fruit", "N");
	ret = iniparser_set(ini, "Cake/Chocolate:Candle_Count", "3");
	ret = iniparser_set(ini, "Cake/Chocolate:Diameter", "20.5");
	ret = iniparser_set(ini, "Cake/Chocolate:Height", "10cm");

	ret = iniparser_set(ini, "Cake/Fruit", NULL);  // caution !!!: it's necessary
	ret = iniparser_set(ini, "Cake/Fruit:Chocolate", "N");
	ret = iniparser_set(ini, "Cake/Fruit:Fruit", "Y");
	ret = iniparser_set(ini, "Cake/Fruit:Candle_Count", "5");
	ret = iniparser_set(ini, "Cake/Fruit:Diameter", "25.0");
	ret = iniparser_set(ini, "Cake/Fruit:Height", "8cm");

	const int nsec = iniparser_getnsec(ini);

	//
	{
		FILE *fp = fopen("data/application_configuration/iniparser/result2.ini", "w");
		iniparser_dump(ini, fp);
		fclose(fp);
	}

	//
	{
		FILE *fp = fopen("data/application_configuration/iniparser/result3.ini", "w");
		iniparser_dump_ini(ini, fp);
		fclose(fp);
	}

	iniparser_freedict(ini);

	std::cin.get();
	return 0 ;
}

}  // namespace local
}  // unnamed namespace

namespace my_iniparser {

}  // namespace my_iniparser

int iniparser_main(int argc, char *argv[])
{
	int	status;

	if (argc < 2)
	{
		local::create_example_ini_file();
		status = local::parse_ini_file("data/application_configuration/iniparser/example.ini");
	}
	else
	{
		status = local::parse_ini_file(argv[1]);
	}

	return status;
}
