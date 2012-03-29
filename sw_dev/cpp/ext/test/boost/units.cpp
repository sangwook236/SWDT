#include <boost/units/systems/si/energy.hpp>
#include <boost/units/systems/si/force.hpp>
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/electric_potential.hpp>
#include <boost/units/systems/si/current.hpp>
#include <boost/units/systems/si/resistance.hpp>
#include <boost/units/systems/si/io.hpp>
#include <boost/units/pow.hpp>
#include <boost/typeof/std/complex.hpp>
#include <complex>
#include <iostream>


namespace {
namespace local {

//>>>>> step #1: base dimension
/// base dimension of length
struct length_base_dimension : boost::units::base_dimension<length_base_dimension, 1> { };
/// base dimension of mass
struct mass_base_dimension : boost::units::base_dimension<mass_base_dimension, 2> { };
/// base dimension of time
struct time_base_dimension : boost::units::base_dimension<time_base_dimension, 3> { };

//>>>>> step #2: dimension
typedef length_base_dimension::dimension_type    length_dimension;
typedef mass_base_dimension::dimension_type      mass_dimension;
typedef time_base_dimension::dimension_type      time_dimension;

typedef boost::units::derived_dimension<length_base_dimension, 2>::type                                                     area_dimension;
typedef boost::units::derived_dimension<mass_base_dimension, 1, length_base_dimension, 2, time_base_dimension, -2>::type    energy_dimension;

//>>>>> step #3: base unit
struct meter_base_unit : boost::units::base_unit<meter_base_unit, length_dimension, 1> { };
struct kilogram_base_unit : boost::units::base_unit<kilogram_base_unit, mass_dimension, 2> { };
struct second_base_unit : boost::units::base_unit<second_base_unit, time_dimension, 3> { };

//>>>>> step #4: unit system
typedef boost::units::make_system<meter_base_unit, kilogram_base_unit, second_base_unit>::type mks_system;

//>>>>> step #5: unit
/// unit typedefs
typedef boost::units::unit<boost::units::dimensionless_type, mks_system>      dimensionless;

typedef boost::units::unit<length_dimension, mks_system>        length;
typedef boost::units::unit<mass_dimension, mks_system>          mass;
typedef boost::units::unit<time_dimension, mks_system>          time;

typedef boost::units::unit<area_dimension, mks_system>          area;
typedef boost::units::unit<energy_dimension, mks_system>        energy;

//>>>>> step #6: unit constant
/// unit constants
BOOST_UNITS_STATIC_CONSTANT(meter, length);
BOOST_UNITS_STATIC_CONSTANT(meters, length);
BOOST_UNITS_STATIC_CONSTANT(kilogram, mass);
BOOST_UNITS_STATIC_CONSTANT(kilograms, mass);
BOOST_UNITS_STATIC_CONSTANT(second, time);
BOOST_UNITS_STATIC_CONSTANT(seconds, time);

BOOST_UNITS_STATIC_CONSTANT(square_meter, area);
BOOST_UNITS_STATIC_CONSTANT(square_meters, area);
BOOST_UNITS_STATIC_CONSTANT(joule, energy);
BOOST_UNITS_STATIC_CONSTANT(joules, energy);

void units_unit()
{
    const length    L;
    const mass      M;
    // needs to be namespace-qualified because of global time definition
    const time      T;
    const energy    E;

    std::cout << "L             = " << L << std::endl
              << "L+L           = " << L + L << std::endl
              << "L-L           = " << L - L << std::endl
              << "L/L           = " << L / L << std::endl
              << "meter*meter   = " << meter * meter << std::endl
              << "M*(L/T)*(L/T) = " << M * (L/T) * (L/T) << std::endl
              << "M*(L/T)^2     = " << M * boost::units::pow<2>(L/T) << std::endl
              << "L^3           = " << boost::units::pow<3>(L) << std::endl
              << "L^(3/2)       = " << boost::units::pow<boost::units::static_rational<3,2> >(L) << std::endl
              << "M^(1/2)       = " << boost::units::root<2>(M) << std::endl
              << "M^(2/3)       = " << boost::units::root<boost::units::static_rational<3,2> >(M) << std::endl;
}

void units_quantity()
{
	{
		boost::units::quantity<length> L = 2.0 * meters;  // quantity of length
		boost::units::quantity<energy> E = kilograms * boost::units::pow<2>(L/seconds);  // quantity of energy

		std::cout << "L                                 = " << L << std::endl
			<< "L+L                               = " << L + L << std::endl
			<< "L-L                               = " << L - L << std::endl
			<< "L*L                               = " << L * L << std::endl
			<< "L/L                               = " << L / L << std::endl
			<< "L*meter                           = " << L * meter << std::endl
			<< "kilograms*(L/seconds)*(L/seconds) = " << kilograms * (L/seconds) * (L/seconds) << std::endl
			<< "kilograms*(L/seconds)^2           = " << kilograms * boost::units::pow<2>(L/seconds) << std::endl
			<< "L^3                               = " << boost::units::pow<3>(L) << std::endl
			<< "L^(3/2)                           = " << boost::units::pow<boost::units::static_rational<3,2> >(L) << std::endl
			<< "L^(1/2)                           = " << boost::units::root<2>(L) << std::endl
			<< "L^(2/3)                           = " << boost::units::root<boost::units::static_rational<3,2> >(L) << std::endl
			<< std::endl;
	}

	{
		boost::units::quantity<length, std::complex<double> > L(std::complex<double>(3.0, 4.0) * meters);
		boost::units::quantity<energy, std::complex<double> > E(kilograms * boost::units::pow<2>(L/seconds));

		std::cout << "L                                 = " << L << std::endl
			<< "L+L                               = " << L + L << std::endl
			<< "L-L                               = " << L - L << std::endl
			<< "L*L                               = " << L * L << std::endl
			<< "L/L                               = " << L / L << std::endl
			<< "L*meter                           = " << L * meter << std::endl
			<< "kilograms*(L/seconds)*(L/seconds) = " << kilograms * (L/seconds) * (L/seconds) << std::endl
			<< "kilograms*(L/seconds)^2           = " << kilograms * boost::units::pow<2>(L/seconds) << std::endl
			<< "L^3                               = " << boost::units::pow<3>(L) << std::endl
			<< "L^(3/2)                           = " << boost::units::pow<boost::units::static_rational<3,2> >(L) << std::endl
			<< "L^(1/2)                           = " << boost::units::root<2>(L) << std::endl
			<< "L^(2/3)                           = " << boost::units::root<boost::units::static_rational<3,2> >(L) << std::endl
			<< std::endl;
	}
}

boost::units::quantity<boost::units::si::energy> work(const boost::units::quantity<boost::units::si::force>& F, const boost::units::quantity<boost::units::si::length>& dx)
{
    return F * dx;
}

void units_si_system()
{
    /// test calcuation of work
    boost::units::quantity<boost::units::si::force> F(2.0 * boost::units::si::newton);
    boost::units::quantity<boost::units::si::length> dx(2.0 * boost::units::si::meter);
    boost::units::quantity<boost::units::si::energy> E(work(F, dx));

    std::cout << "F  = " << F << std::endl
              << "dx = " << dx << std::endl
              << "E  = " << E << std::endl
              << std::endl;

    // check complex quantities
    typedef std::complex<double> complex_type;

    boost::units::quantity<boost::units::si::electric_potential, complex_type> v = complex_type(12.5, 0.0) * boost::units::si::volts;
    boost::units::quantity<boost::units::si::current, complex_type> i = complex_type(3.0, 4.0) * boost::units::si::amperes;
    boost::units::quantity<boost::units::si::resistance, complex_type> z = complex_type(1.5, -2.0) * boost::units::si::ohms;

    std::cout << "V   = " << v << std::endl
              << "I   = " << i << std::endl
              << "Z   = " << z << std::endl
              << "I*Z = " << i * z << std::endl
              << "I*Z == V? " << std::boolalpha << (i * z == v) << std::endl
              << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace boost {
namespace units {

//>>>>> step #7: base unit info
template<> struct base_unit_info<local::meter_base_unit>
{
    static std::string name()               { return "meter"; }
    static std::string symbol()             { return "m"; }
};

template<> struct base_unit_info<local::kilogram_base_unit>
{
    static std::string name()               { return "kilogram"; }
    static std::string symbol()             { return "kg"; }
};

template<> struct base_unit_info<local::second_base_unit>
{
    static std::string name()               { return "second"; }
    static std::string symbol()             { return "s"; }
};

}  // namespace units
}  // unnamed boost

void units()
{
	local::units_unit();
	std::cout << std::endl;
	local::units_quantity();
	std::cout << std::endl;
	local::units_si_system();
}
