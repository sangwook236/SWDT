using System;
using System.Collections.Generic;
using System.Text;

namespace nunit
{
    using NUnit.Framework;

    [TestFixture]
    class AccountTest
    {
        [SetUp]
        public void setUp()
        {
            source_ = new Account();
            source_.Deposit(200.00F);
            destination_ = new Account();
            destination_.Deposit(150.00F);
        }

        [TearDown]
        public void tearDown()
        {
        }

        [Test]
        public void testTransferFunds()
        {
            source_.TransferFunds(destination_, 100.00f);
            Assert.AreEqual(250.00F, destination_.Balance);
            Assert.AreEqual(100.00F, source_.Balance);
        }

        [Test]
        [ExpectedException(typeof(InsufficientFundsException))]
        public void testTransferWithInsufficientFunds()
        {
            source_.TransferFunds(destination_, 300.00F);
        }

        [Test]
        [Ignore("Decide how to implement transaction management")]
        public void testTransferWithInsufficientFundsAtomicity()
        {
            try
            {
                source_.TransferFunds(destination_, 300.00F);
            }
            catch (InsufficientFundsException)
            {
            }

            Assert.AreEqual(200.00F, source_.Balance);
            Assert.AreEqual(150.00F, destination_.Balance);
        }

        Account source_;
        Account destination_;
    }
}
