#include <boost/spirit/include/classic.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

void	do_print_ch(char ch)  {  std::cout << ch << ' ' << std::endl;  }
void	do_print_str(char const *start, char const *end)
{
	std::string str(start, end);
	std::cout << std::string(start, end).c_str() <<  ' ' << std::endl;
}
void	do_print_int(unsigned int ui)  {  std::cout << ui <<  ' ' << std::endl;  }

////////////////////////////////////////////////////////////////////////////
struct CspRuleParser : public boost::spirit::classic::grammar<CspRuleParser>
{
	template <typename ScannerT>
	struct definition
	{
		definition(CspRuleParser const& /*self*/)
		{
			expression_test
				=	rule_expression
				|	variable_spec
				|	constraint_spec
				|	variable_name
				|	domain_values
				|	constraint_type
				|	comment_stmt
				;

			rule_expression
				=	+(	variable_spec
					|	constraint_spec
					|	comment_stmt
					)
				;

			variable_spec
				=	(boost::spirit::classic::ch_p('(') >> boost::spirit::classic::str_p("variable")[&do_print_str] >> variable_name >> domain_values >> boost::spirit::classic::ch_p(')'))[&do_print_str]
				;

			constraint_spec
				=	(boost::spirit::classic::ch_p('(') >> boost::spirit::classic::str_p("constraint")[&do_print_str] >> constraint_type >> variable_name >> variable_name >> boost::spirit::classic::ch_p(')'))[&do_print_str]
				;

			variable_name
				=	boost::spirit::classic::lexeme_d[(boost::spirit::classic::alpha_p >> *boost::spirit::classic::alnum_p)[&do_print_str]]
				;

			domain_values
				=	(boost::spirit::classic::ch_p('(') >> +boost::spirit::classic::uint_p[&do_print_int] >> boost::spirit::classic::ch_p(')'))[&do_print_str]
				;

			constraint_type
				=   boost::spirit::classic::str_p("eq")[&do_print_str]
				|   boost::spirit::classic::str_p("neq")[&do_print_str]
				;

			comment_stmt
				=   (boost::spirit::classic::ch_p('#') >> boost::spirit::classic::lexeme_d[*(boost::spirit::classic::anychar_p - boost::spirit::classic::eol_p)])[&do_print_str]
				;
		}

		boost::spirit::classic::rule<ScannerT> expression_test, rule_expression, variable_spec, constraint_spec, variable_name, domain_values, constraint_type, comment_stmt;

		boost::spirit::classic::rule<ScannerT> const & start() const { return expression_test; }
	};
};

}  // namespace local
}  // unnamed namespace

void spirit_classic()
{
	std::cout << "/////////////////////////////////////////////////////////\n\n";
	std::cout << "\t\tExpression parser...\n\n";
	std::cout << "/////////////////////////////////////////////////////////\n\n";
	std::cout << "Type an expression...or [q or Q] to quit\n\n";

	local::CspRuleParser cspParser;    //  Our parser

	std::string str;
	while (std::getline(std::cin, str))
	{
		if (str.empty() || str[0] == 'q' || str[0] == 'Q')
			break;

		//str = "(variable f1 (3))(constraint neq b1 b2)# it's a comment\n(constraint neq b1 b3)\n(variable c1 (1 2 3 4 5 6 7 8 9))#abcdedf(constraint neq d7 e8)";
		boost::spirit::classic::parse_info<> info = boost::spirit::classic::parse(str.c_str(), cspParser, boost::spirit::classic::space_p);

		if (info.full)
		{
			std::cout << "-------------------------\n";
			std::cout << "Parsing succeeded\n";
			std::cout << "-------------------------\n";
		}
		else
		{
			std::cout << "-------------------------\n";
			std::cout << "Parsing failed\n";
			std::cout << "stopped at: \": " << info.stop << "\"\n";
			std::cout << "-------------------------\n";
		}
	}

	std::cout << "Bye... :-) \n\n";
}
