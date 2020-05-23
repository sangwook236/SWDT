#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://mimesis.readthedocs.io/

from mimesis import Generic, Person, Address, Business, Payment, Numbers
from mimesis import locales
from mimesis.enums import Gender
from mimesis.builtins import USASpecProvider, BrazilSpecProvider

# REF [site] >> https://mimesis.readthedocs.io/getting_started.html
def getting_started_example():
	generic = Generic()
	#generic = Generic(locales.EN)

	print('Month =', generic.datetime.month())
	print('Datetime =', generic.datetime.datetime(start=1900, end=2035, timezone=None))  # Type: datetime.datetime.
	print('IMEI =', generic.code.imei())
	print('Fruit =', generic.food.fruit())
	print('RNA =', generic.science.rna_sequence())

	print('Word =', generic.text.word())

	with generic.text.override_locale(locales.FR):
		print('Word =', generic.text.word())

	print('Word =', generic.text.word())

	generic = Generic('en')
	generic.add_provider(USASpecProvider)

	print('SSN =', generic.usa_provider.ssn())
	#print('CPF =', generic.usa_provider.cpf())  # AttributeError: 'USASpecProvider' object has no attribute 'cpf'.

	generic = Generic('pt-br')
	#generic = Generic(locales.PT_BR)
	generic.add_provider(BrazilSpecProvider)

	#print('SSN =', generic.brazil_provider.ssn())  # AttributeError: 'BrazilSpecProvider' object has no attribute 'ssn'.
	print('CPF =', generic.brazil_provider.cpf())

	#--------------------
	numbers = Numbers()

	print('Numbers =', numbers.between())  # Type: int.
	print('Numbers =', numbers.between(10, 10000000000000000))  # Type: int.

	#--------------------
	person = Person(locales.KO)

	print('Full name =', person.full_name(gender=Gender.FEMALE))
	print('Full name =', person.full_name(gender=Gender.MALE, reverse=True))

	with person.override_locale(locales.RU):
		print('Full name =', person.full_name())

	print('Full name =', person.full_name())
	print('Telephone =', person.telephone())
	print('Telephone =', person.telephone(mask='(###)-###-####'))
	print('Identifier =', person.identifier())
	print('Identifier =', person.identifier(mask='######-#######'))

	#--------------------
	de = Address('de')
	ru = Address('ru')

	print('Region =', de.region())
	print('Federal subject =', ru.federal_subject())
	print('Address =', de.address())
	print('Address =', ru.address())

	ko = Address('ko')

	print('Address =', ko.province(), ko.city(), ko.address())
	print('Zip code =', ko.zip_code())

	#--------------------
	business = Business('ko')
	
	#print('Price =', business.price(minimum=1.0, maximum=1000000000.0))  # Type: str.
	#print('Price =', business.price(minimum=1.0, maximum=1000000000.0)[:-2])  # Type: str.
	print('Price =', business.price(minimum=1.0, maximum=1000000000.0)[:-5])  # Type: str.

	#--------------------
	payment = Payment()
	
	print('Credit card =', payment.credit_card_number(card_type=None))  # Type: str.

def main():
	getting_started_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
