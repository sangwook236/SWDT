using System;
using System.Collections.Generic;
using System.Text;

namespace nunit
{
    public class Account
    {
        public void Deposit(float amount)
        {
            balance_ += amount;
        }

        public void Withdraw(float amount)
        {
            balance_ -= amount;
        }

        public void TransferFunds(Account destination, float amount)
        {
            if (balance_ - amount < minimumBalance_)
                throw new InsufficientFundsException();
            destination.Deposit(amount);
            Withdraw(amount);
        }

        public float Balance
        {
            get { return balance_; }
        }

        public float MinimumBalance
        {
            get { return minimumBalance_; }
        }
        
        private float balance_;
        private float minimumBalance_ = 10.00F;
    }

    public class InsufficientFundsException: ApplicationException
    {
    }
}
