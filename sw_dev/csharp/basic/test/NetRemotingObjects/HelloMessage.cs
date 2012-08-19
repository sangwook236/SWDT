using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NetRemotingObjects
{
    [Serializable]
    public class HelloMessage
    {
        public HelloMessage(string msg)
        {
            msg_ = msg;
        }

        public string Message
        {
            get { return msg_; }
        }

        private string msg_;
    }
}
