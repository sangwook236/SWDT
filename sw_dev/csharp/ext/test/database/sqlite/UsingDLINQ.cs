using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace database
{
    using System.Data;
    //using System.Data.Linq;
    //using DbLinq.Data.Linq;
    using System.Data.SQLite;

    class UsingDLINQ
    {
        public static void runTests()
        {
#if false
            Console.WriteLine(">>>>> after inserting ...");
            runInsertOperation1();
            runSelectOperation1();

            Console.WriteLine("\n>>>>> after updating ...");
            runUpdateOperation1();
            runSelectOperation1();

            Console.WriteLine("\n>>>>> after deleting ...");
            runDeleteOperation1();
            runSelectOperation1();
#else
            Console.WriteLine(">>>>> after inserting ...");
            runInsertOperation2();
            runSelectOperation2();

            Console.WriteLine("\n>>>>> after updating ...");
            runUpdateOperation2();
            runSelectOperation2();

            Console.WriteLine("\n>>>>> after deleting ...");
            runDeleteOperation2();
            runSelectOperation2();
#endif
        }

        static void runSelectOperation1()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (DbLinq.Data.Linq.DataContext dbContext = new DbLinq.Data.Linq.DataContext(connection))
                    {
                        DbLinq.Data.Linq.Table<Color> colors = dbContext.GetTable<Color>();
                        var queriedColors = from c in colors
                                            select c;

                        foreach (var color in queriedColors)
                            Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", color.ID, color.Name, color.HexChars, (null == color.Description ? "null" : color.Description), (null == color.HexCode ? "null" : string.Format("0x{0:X}", color.HexCode)));
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runSelectOperation2()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (ColorDataContext dbContext = new ColorDataContext(connection))
                    {
                        //Color[] colors = dbContext.Colors.ToArray();
                        //List<Color> colors = dbContext.Colors.ToList();

                        var queriedColors = from c in dbContext.Colors
                                            select c;

                        foreach (var color in queriedColors)
                            Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", color.ID, color.Name, color.HexChars, (null == color.Description ? "null" : color.Description), (null == color.HexCode ? "null" : string.Format("0x{0:X}", color.HexCode)));
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runInsertOperation1()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (DbLinq.Data.Linq.DataContext dbContext = new DbLinq.Data.Linq.DataContext(connection))
                    {
                        DbLinq.Data.Linq.Table<Color> colors = dbContext.GetTable<Color>();

                        //Color magenta = new Color { Name = "megenta", HexChars = "ff00ff", Description = null, HexCode = null };
                        Color magenta = new Color { Name = "megenta", HexChars = "ff00ff", Description = "it is a magenta", HexCode = 0xFF00FF };
                        colors.InsertOnSubmit(magenta);

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }

            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (DbLinq.Data.Linq.DataContext dbContext = new DbLinq.Data.Linq.DataContext(connection))
                    {
                        DbLinq.Data.Linq.Table<Color> colors = dbContext.GetTable<Color>();

                        for (int i = 0; i < 16; ++i)
                            colors.InsertOnSubmit(new Color() { Name = string.Format("Color #{0:D2}", i + 1), HexChars = string.Format("{0:X6}", i + 1) });

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runInsertOperation2()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (ColorDataContext dbContext = new ColorDataContext(connection))
                    {
                        //Color magenta = new Color() { Name = "magenta", HexChars = "ff00ff", Description = null, HexCode = null };
                        Color magenta = new Color { Name = "magenta", HexChars = "ff00ff", Description = "it is a magenta", HexCode = 0xFF00FF };
                        dbContext.Colors.InsertOnSubmit(magenta);

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }

            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (ColorDataContext dbContext = new ColorDataContext(connection))
                    {
                        for (int i = 0; i < 16; ++i)
                        {
                            Color color = new Color() { Name = string.Format("Color #{0:D2}", i + 1), HexChars = string.Format("{0:X6}", i + 1) };
                            dbContext.Colors.InsertOnSubmit(color);
                        }

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runUpdateOperation1()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (DbLinq.Data.Linq.DataContext dbContext = new DbLinq.Data.Linq.DataContext(connection))
                    {
                        DbLinq.Data.Linq.Table<Color> colors = dbContext.GetTable<Color>();

                        // for an existing record.
                        {
                            try
                            {
                                Color queriedColor = (from c in colors
                                                       //where string.Equals(c.Name, "Color #01")  // not working
                                                       where c.Name == "Color #01"
                                                       select c).First();
                                                       //select c).Single();

                                queriedColor.HexChars = "0F0F0F";
                            }
                            catch (InvalidOperationException ex)
                            {
                                Console.WriteLine("DB query error: {0}", ex.Message);
                            }

                            try
                            {
                                dbContext.SubmitChanges();
                            }
                            catch (System.Data.Linq.ChangeConflictException)
                            {
                                foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                                {
                                    foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                    {
                                        Console.WriteLine("Conflict Details:");
                                        Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                        Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                        Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                    }

                                    conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                                }
                            }
                        }

                        // a new record is not inserted.
                        {
                            try
                            {
                                Color queriedColor = (from c in colors
                                                       //where string.Equals(c.Name, "Color #20")  // not working
                                                       where c.Name == "Color #20"
                                                       select c).First();
                                                       //select c).Single();

                                queriedColor.HexChars = "0F0F0F";
                            }
                            catch (InvalidOperationException ex)
                            {
                                Console.WriteLine("DB query error: {0}", ex.Message);
                            }

                            try
                            {
                                dbContext.SubmitChanges();
                            }
                            catch (System.Data.Linq.ChangeConflictException)
                            {
                                foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                                {
                                    foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                    {
                                        Console.WriteLine("Conflict Details:");
                                        Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                        Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                        Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                    }

                                    conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                                }
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runUpdateOperation2()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (ColorDataContext dbContext = new ColorDataContext(connection))
                    {
                        // for an existing record.
                        {
                            try
                            {
                                Color queriedColor = (from c in dbContext.Colors
                                                       //where string.Equals(c.Name, "Color #01")  // not work
                                                       where c.Name == "Color #01"
                                                       //select c).First();
                                                       select c).Single();

                                queriedColor.HexChars = "0F0F0F";
                            }
                            catch (InvalidOperationException ex)
                            {
                                Console.WriteLine("DB query error: {0}", ex.Message);
                            }

                            try
                            {
                                dbContext.SubmitChanges();
                            }
                            catch (System.Data.Linq.ChangeConflictException)
                            {
                                foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                                {
                                    foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                    {
                                        Console.WriteLine("Conflict Details:");
                                        Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                        Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                        Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                    }

                                    conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                                }
                            }
                        }

                        // a new record is not inserted.
                        {
                            try
                            {
                                Color queriedColor = (from c in dbContext.Colors
                                                       //where string.Equals(c.Name, "Color #20")  // not working
                                                       where c.Name == "Color #20"
                                                       //select c).First();
                                                       select c).Single();

                                queriedColor.HexChars = "OFOFOF";
                            }
                            catch (InvalidOperationException ex)
                            {
                                Console.WriteLine("DB query error: {0}", ex.Message);
                            }

                            try
                            {
                                dbContext.SubmitChanges();
                            }
                            catch (System.Data.Linq.ChangeConflictException)
                            {
                                foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                                {
                                    foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                    {
                                        Console.WriteLine("Conflict Details:");
                                        Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                        Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                        Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                    }

                                    conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                                }
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runDeleteOperation1()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (DbLinq.Data.Linq.DataContext dbContext = new DbLinq.Data.Linq.DataContext(connection))
                    {
                        DbLinq.Data.Linq.Table<Color> colors = dbContext.GetTable<Color>();

                        var queriedColors = from c in colors
                                            where c.ID > 3
                                            select c;

                        foreach (Color color in queriedColors)
                            colors.DeleteOnSubmit(color);

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runDeleteOperation2()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    using (ColorDataContext dbContext = new ColorDataContext(connection))
                    {
                        var queriedColors = from c in dbContext.Colors
                                            where c.ID > 3
                                            select c;

                        foreach (Color color in queriedColors)
                            dbContext.Colors.DeleteOnSubmit(color);

                        try
                        {
                            dbContext.SubmitChanges();
                        }
                        catch (System.Data.Linq.ChangeConflictException)
                        {
                            foreach (System.Data.Linq.ObjectChangeConflict conflict in dbContext.ChangeConflicts)
                            {
                                foreach (System.Data.Linq.MemberChangeConflict changeConflict in conflict.MemberConflicts)
                                {
                                    Console.WriteLine("Conflict Details:");
                                    Console.WriteLine("\tOriginal value retrieved from database: {0}", changeConflict.OriginalValue);
                                    Console.WriteLine("\tCurrent value in database: {0}", changeConflict.DatabaseValue);
                                    Console.WriteLine("\tOriginal value in memory: {0}", changeConflict.CurrentValue);
                                }

                                conflict.Resolve(System.Data.Linq.RefreshMode.OverwriteCurrentValues);
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        private static string connectionStr_ = "DbLinqProvider=Sqlite;" + "Data Source=..\\data\\database\\sqlite\\sqlite3_test.db;Version=3;";
        //private static string connectionStr_ = Properties.Settings.Default.TestDatabase;
    }
}
