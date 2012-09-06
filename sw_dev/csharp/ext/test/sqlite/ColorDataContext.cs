using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite
{
    using System.Data;
    //using System.Data.Linq;
    using System.Data.Linq.Mapping;
    //using System.Data.SQLite;

    class ColorDataContext : DbLinq.Data.Linq.DataContext
    {
        public ColorDataContext(IDbConnection connection)
            : base(connection)
        {
        }

        public ColorDataContext(IDbConnection connection, MappingSource mapping)
            : base(connection, mapping)
        {
        }

        public ColorDataContext(string fileOrServerOrConnection)
            : base(fileOrServerOrConnection)
        {
        }

        public ColorDataContext(string fileOrServerOrConnection, MappingSource mapping)
            : base(fileOrServerOrConnection, mapping)
        {
        }

        public DbLinq.Data.Linq.Table<Color> Colors
        {
            get { return GetTable<Color>(); }
        }
    }
}
