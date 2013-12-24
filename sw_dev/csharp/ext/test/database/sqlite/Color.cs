using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace database.sqlite
{
    using System.Data;
    //using System.Data.Linq;
    using System.Data.Linq.Mapping;
    //using DbLinq.Data.Linq;
    //using System.Data.SQLite;

    [Table(Name = "Colors")]
    class Color
    {
        [Column(Name = "id", IsPrimaryKey = true, IsDbGenerated = true, UpdateCheck = UpdateCheck.Never, CanBeNull = false)]
        //[Column(Name = "id", DbType = "BIGINT NOT NULL IDENTITY", IsPrimaryKey = true, IsDbGenerated = true, UpdateCheck = UpdateCheck.Never, CanBeNull = false)]
        public long ID { get; set; }

        [Column(Name = "name", CanBeNull = false)]
        //[Column(Name = "name", DbType = "VARCHAR(128) NOT NULL", CanBeNull = false)]
        public string Name { get; set; }

        [Column(Name = "hexchars", CanBeNull = false)]
        //[Column(Name = "hexchars", DbType = "VARCHAR(6) NOT NULL", CanBeNull = false)]
        public string HexChars { get; set; }

        [Column(Name = "description", CanBeNull = true)]
        //[Column(Name = "description", DbType = "VARCHAR(128) NULL", CanBeNull = true)]
        public string Description { get; set; }

        [Column(Name = "hexcode", CanBeNull = true)]
        //[Column(Name = "hexcode", DbType = "BIGINT NULL", CanBeNull = true)]
        public long? HexCode { get; set; }  // caution: the type is long?, but not long.
    }
}
