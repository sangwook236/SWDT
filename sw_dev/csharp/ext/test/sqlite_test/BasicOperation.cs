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
            runInsertOperation();
            runSelectOperation();
            runDeleteOperation();
        }

        static void runInsertOperation()
        {
            using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
            {
                connection.Open();

                using (SQLiteTransaction transaction = connection.BeginTransaction())
                {
                    using (SQLiteCommand command = new SQLiteCommand(connection))
                    {
                        //string sql = String.Format("INSERT INTO Colors(id, name, hex) VALUES({0}, '{1}', '{2}')", 4, "magenta", "ff00ff");  // it's also working
                        string sql = String.Format("INSERT INTO Colors(name, hex) VALUES('{0}', '{1}')", "magenta", "ff00ff");

                        command.CommandText = sql;
                        command.ExecuteNonQuery();
                    }

                    transaction.Commit();
                }
            }

            using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
            {
                connection.Open();

                using (SQLiteTransaction transaction = connection.BeginTransaction())
                {
                    using (SQLiteCommand command = new SQLiteCommand(connection))
                    {
                        SQLiteParameter param1 = new SQLiteParameter();
                        SQLiteParameter param2 = new SQLiteParameter();

                        string sql = "INSERT INTO Colors(name, hex) VALUES(?, ?)";

                        command.CommandText = sql;
                        command.Parameters.Add(param1);
                        command.Parameters.Add(param2);

                        for (int i = 0; i < 16; ++i)
                        {
                            param1.Value = String.Format("Color #{0:D2}", i + 1);
                            param2.Value = String.Format("{0:X6}", i + 1);
                            command.ExecuteNonQuery();
                        }
                    }

                    transaction.Commit();
                }
            }
        }

        static void runSelectOperation()
        {
            using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
            {
                connection.Open();

                string sql = "SELECT * FROM Colors";
                using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                {
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

        static void runDeleteOperation()
        {
            using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
            {
                connection.Open();

                using (SQLiteTransaction transaction = connection.BeginTransaction())
                {
                    using (SQLiteCommand command = new SQLiteCommand(connection))
                    {
                        string sql = "DELETE FROM Colors WHERE id > 3";

                        command.CommandText = sql;
                        command.ExecuteNonQuery();
                    }

                    transaction.Commit();
                }
             }
        }

        private static string connectionStr_ = @"Data Source=""..\data\sqlite_data\sqlite3_test.db"";Version=3;";
    }
}
