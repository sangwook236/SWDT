using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite_test
{
    using System.Data.SQLite;
    using System.Data;

    class BasicOperation
    {
        public static void run()
        {
            runBasicOperation1();
        }

        static void runBasicOperation1()
        {
            string connectionStr = @"Data Source=""..\data\sqlite_data\sqlite3_test.db"";Version=3;";
            string sql = "select * from Colors";
            using (SQLiteConnection connection = new SQLiteConnection(connectionStr))
            {
                SQLiteCommand cmd = new SQLiteCommand(sql, connection);

                connection.Open();

                SQLiteDataReader reader = null;
                try
                {
                    reader = cmd.ExecuteReader(CommandBehavior.CloseConnection);
                    while (reader.Read())
                    {
                        Console.WriteLine(reader["name"] + " " + reader["hex"]);
                    }
                }
                finally
                {
                    reader.Close();
                }
            }
        }
    }
}
