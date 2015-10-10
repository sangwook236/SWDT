// Probabilistic Reasoning Library (PRL)
// Copyright (C) 2005  Mark Andrew Paskin
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef PRL_LOG_SPACE_HPP
#define PRL_LOG_SPACE_HPP

#include <limits>
#include <iostream>
#include <cmath>
#include <functional>

namespace prl {

  //! A tag that is used to indicate a log-space value.
  struct log_tag_t { };

  /**
   * A numeric representation which represents the real number \f$x\f$
   * using a floating point representation of \f$\log x\f$.  This
   * class is designed so that the representation of the value is
   * transparent to the user.  E.g., all operators that are defined
   * for double representations are also defined for log_t<double>
   * representations, and they have the same meaning.  These operators
   * can also be used with both types simultaneously, e.g., 1.0 +
   * log_t<double>(2.0) == 3.0.
   *
   */
  template <typename storage_t = double>
  class log_t {

  protected:

    /**
     * The log space representation of \f$x\f$, i.e., the value
     * \f$\log x\f$.
     */
    storage_t lv;

  public:

    /**
     * Default constructor.  The value is initialized to represent
     * zero.
     */
    log_t() : lv(-std::numeric_limits<storage_t>::infinity()) { }

    /**
     * Log-space constructor. 
     *
     * @param lv the logarithm of the value this object should 
     *           represent
     */
    log_t(const storage_t& lv, log_tag_t) : lv(lv) { }

    /**
     * Constructor.  Note that the parameter is the value to be
     * represented, not its logarithm.
     *
     * @param value the value this object should represent
     */
    log_t(const storage_t& value) { 
      if (value == static_cast<storage_t>(0))
	lv = -std::numeric_limits<storage_t>::infinity();
      else
	lv = log(value);
    }

    //! Copy constructor.
    log_t(const log_t& a) : lv(a.lv) { }

    /**
     * Assignment.  This value is reset to the supplied value.
     *
     * @param a the log-space representation of \f$x\f$
     * @return  this value, after it is updated to also represent 
     *          \f$x\f$ in log-space
     */
    inline const log_t& operator=(const log_t& a) { 
      this->lv = a.lv;
      return *this;
    }

    /**
     * Assignment into log space.  Note that the parameter is not in
     * log space.
     *
     * @param a the value \f$x\f$
     * @return  this value, after it is updated to represent \f$x\f$ 
     *          in log space
     */
    inline const log_t& operator=(const storage_t& x) { 
      if (x == static_cast<storage_t>(0))
	this->lv = -std::numeric_limits<storage_t>::infinity();
      else
	this->lv = log(x);
      return *this;
    }

    /**
     * Conversion out of log space.  Casting a log-space value into
     * its associated storage type computes the standard
     * representation from the log-space representation.
     *
     * @return  the value \f$x\f$, where this object represents
     *          \f$x\f$ in log-space
     */
    inline operator storage_t() const { 
      return exp(this->lv);
    }

    /**
     * Returns the internal representation of the value in log space.
     *
     * @return  the value \f$\log x\f$, where this object represents
     *          \f$x\f$ in log-space
     */
    inline storage_t get_log_value() const { 
      return this->lv;
    }

    /**
     * Returns the log space value representing the sum of this
     * value and the supplied log space value.
     *
     * This routine exploits a special purpose algorithm called log1p
     * that is in the C standard library.  log1p(x) computes the value
     * \f$\log(1 + x)\f$ in a numerically stable way when \f$x\f$ is
     * close to zero.  Note that
     * \[
     *  \log(1 + y/x) + \log(x) = \log(x + y)
     * \]
     * Further note that 
     * \[
     *  y/x = \exp(\log y - \log x)
     * \]
     * Thus, we first compute \f$y/x\f$ stably by choosing \f$x >
     * y\f$, and then use log1p to implement the first equation.
     *
     * @param a the value \f$\log x\f$
     * @return  the value \f$\log (x + y)\f$, where this object
     *          represents \f$\log y\f$
     */
    inline log_t operator+(const log_t& a) const {
      if (a.lv == -std::numeric_limits<storage_t>::infinity())
	return *this;
      if (this->lv == -std::numeric_limits<storage_t>::infinity())
	return a;
      storage_t lx, ly;
      if (this->lv < a.lv) {
	lx = a.lv;
	ly = this->lv;
      } else {
	lx = this->lv;
	ly = a.lv;
      }
      return log_t(log1p(exp(ly - lx)) + lx, prl::log_tag_t());
    }

    /**
     * Computes the sum of this (log-space) value and the supplied
     * value.
     *
     * @param  y the value \f$y\f$ to be added to this (log-space) 
     *           value 
     * @return a log-space representation of \f$x + y\f$, where this
     *           object represents \f$x\f$ in log-space
     */
    inline log_t operator+(const storage_t& y) const {
      return *this + log_t(y);
    }

    /**
     * Updates this object to represent the sum of this value and the
     * supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator+
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x + y\f$ in log-space, where this object originally
     *          represented \f$x\f$ in log-space
     */
    template <typename representation_t>
    inline const log_t& operator+=(const representation_t& y) {
      *this = *this + y;
      return *this;
    }

    /**
     * Returns the value representing the product of this value and
     * the supplied value.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  the value \f$x \times y\f$ represented in log-space, 
     *          where this object represents \f$x\f$ in log-space
     */
    inline log_t operator*(const log_t& a) const {
      return log_t(this->lv + a.lv, log_tag_t());
    }

    /**
     * Returns the value representing the product of this value and
     * the supplied value.
     *
     * @param y the value \f$y\f$
     * @return  the value \f$x \times y\f$ represented in log-space, 
     *          where this object represents \f$x\f$ in log-space
     */
    inline log_t operator*(const storage_t& y) const {
      return (*this) * log_t(y);
    }

    /**
     * Updates this object to represent the product of this value and
     * the supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator*
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x \times y\f$ in log-space, where this object 
     *          originally represented \f$x\f$ in log-space
     */
    template <typename representation_t>
    inline const log_t& operator*=(const representation_t& y) {
      (*this) = (*this) * y;
      return *this;
    }

    /**
     * Returns the value representing the ratio of this value and the
     * supplied value.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  the value \f$x / y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline log_t operator/(const log_t& a) const {
      return log_t(this->lv - a.lv, log_tag_t());
    }

    /**
     * Returns the value representing the ratio of this value and the
     * supplied value.
     *
     * @param y the value \f$y\f$
     * @return  the value \f$x / y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline log_t operator/(const storage_t& y) const {
      return (*this) / log_t(y);
    }

    /**
     * Updates this object to represent the ratio of this value and
     * the supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator/
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x \times y\f$ in log-space, where this object 
     *          originally represented \f$x\f$ in log-space
     */
    template <typename representation_t>
    inline const log_t& operator/=(const representation_t& y) {
      (*this) = (*this) / y;
      return *this;
    }

    /**
     * Returns true iff this object represents the same value as the
     * supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x = y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator==(const log_t& a) const {
      return (this->lv == a.lv);
    }

    /**
     * Returns true iff this object represents a different value from
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \neq y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator!=(const log_t& a) const {
      return (this->lv != a.lv);
    }

    /**
     * Returns true iff this object represents a smaller value than
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x < y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator<(const log_t& a) const {
      return (this->lv < a.lv);
    }

    /**
     * Returns true iff this object represents a larger value than
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x > y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator>(const log_t& a) const {
      return (this->lv > a.lv);
    }

    /**
     * Returns true iff this object represents a value that is less
     * than or equal to the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \le y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator<=(const log_t& a) const {
      return (this->lv <= a.lv);
    }

    /**
     * Returns true iff this object represents a value that is greater
     * than or equal to the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \ge y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    inline bool operator>=(const log_t& a) const {
      return (this->lv >= a.lv);
    }

    /**
     * Writes this log space representation to the supplied stream.
     */
    template <typename char_t, 
	      typename traits_t>
    void write(typename std::basic_ostream<char_t, traits_t>& out) const {
      out << "exp(" << lv << ")";
    }

    /**
     * Reads this log space value from the supplied stream.  There are
     * two accepted formats.  The first the same format used by the
     * storage_t type; numbers in this format are converted into log
     * space representation.  The second format is 'exp(X)', where X
     * is in a format used by the storage_t type.  In this case, the
     * read value is treated as a log space value.  For example,
     * reading '1.23e4' causes this object to represent the value
     * 1.23e4 in log space, as log(1.23e4); reading the value
     * exp(-1234.5) causes this object to represent the value
     * \f$e^{-1234.5}\f$, by storing the value 1234.5.
     *
     * @param in the stream from which this value is read
     */
    template <typename char_t, 
	      typename traits_t>
    void read(typename std::basic_istream<char_t, traits_t>& in) {
      // Read off any leading whitespace.
      in >> std::ws;
      // Check to see if this value is written in log space.
      typedef typename std::basic_istream<char_t, traits_t>::int_type int_t;
      if (in.peek() == static_cast<int_t>('e')) {
	in.ignore(4);
	in >> lv;
	in.ignore(1);
      } else {
	storage_t x;
	in >> x;
	*this = x;
      }
    }

  }; // struct log_t<storage_t>

  /**
   * Returns the value representing the sum of this value and the
   * supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x + y\f$ represented in log-space
   */
  template <typename storage_t>
  inline log_t<storage_t> operator+(const storage_t& x, 
				    const log_t<storage_t>& a) { 
    return a + x; 
  }

  /**
   * Returns the value representing the product of this value and
   * the supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x \times y\f$ represented in log-space
   */
  template <typename storage_t>
  inline log_t<storage_t> operator*(const storage_t& x, 
				    const log_t<storage_t>& a) { 
    return a * x; 
  }

  /**
   * Returns the value representing the ratio of this value and
   * the supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x / y\f$ represented in log-space
   */
  template <typename storage_t>
  inline log_t<storage_t> operator/(const storage_t& x, 
				    const log_t<storage_t>& a) { 
    return log_t<storage_t>(x) / a; 
  }

} // end of namespace: prl

/**
 * Writes a log space value to a stream.
 */
template <typename char_t, 
	  typename traits_t,
	  typename storage_t>
typename std::basic_ostream<char_t, traits_t>&
operator<<(typename std::basic_ostream<char_t, traits_t>& out, 
	   const typename prl::log_t<storage_t>& x) {
  x.write(out);
  return out;
}

/**
 * Reads a log space value from a stream.
 */
template <typename char_t, 
	  typename traits_t,
	  typename storage_t>
typename std::basic_istream<char_t, traits_t>&
operator>>(typename std::basic_istream<char_t, traits_t>& in, 
	   typename prl::log_t<storage_t>& x) {
  x.read(in);
  return in;
}

#endif // #ifndef PRL_LOG_SPACE_HPP
