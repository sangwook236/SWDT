using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace TestApp
{
    class Collection
    {
        public static void run()
        {
			Console.WriteLine(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> List");
			runList();
			Console.WriteLine("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HashSet");
			runHashSet();
			Console.WriteLine("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dictionary");
			runDictionary();
			Console.WriteLine("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SortedList");
			runSortedList();
        }

        struct Pair
        {
            public Pair(int key, int val)
            {
                this.key = key;
                this.val = val;
            }

            public Pair(Pair rhs)
            {
                this.key = rhs.key;
                this.val = rhs.val;
            }

            public int key;
            public int val;
        }

        class PairComparer : IComparer<Pair>
        {
            int IComparer<Pair>.Compare(Pair lhs, Pair rhs)
            //public int Compare(Pair lhs, Pair rhs)
            {
                return lhs.val < rhs.val ? -1 : lhs.val > rhs.val ? 1 : 0;  // ascending order
                //return lhs.val < rhs.val ? 1 : lhs.val > rhs.val ? -1 : 0;  // descending order
            }
        }

		class PairEqualityComparer : IEqualityComparer<Pair>
		{
			bool IEqualityComparer<Pair>.Equals(Pair lhs, Pair rhs)
			{
				return lhs.key == rhs.key;
			}

			int IEqualityComparer<Pair>.GetHashCode(Pair obj)
			{
				//return obj.GetHashCode();
				return obj.key;
			}
		}

        class KeyComparer : IComparer<int>
        {
            public int Compare(int lhs, int rhs)
            {
                //return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;  // ascending order
                return lhs < rhs ? 1 : lhs > rhs ? -1 : 0;  // descending order
            }
        }

        static void runList()
        {
            {
                List<int> list = new List<int>();
                try
                {
                    list.Add(3);
                    list.Add(3);
                    list.Add(1);
                    list.Add(5);
                    list.Add(4);
                    list.Add(4);
                    list.Add(3);
                    list.Add(1);
                    list.Add(5);
                    list.Add(4);
                    list.Add(1);
                }
                catch (ArgumentException)
                {
                    Console.WriteLine("ArgumentException occurred !!!");
                }

                Console.Write("elements = ");
                foreach (int entry in list)
                    Console.Write(entry + " ");
                Console.WriteLine();

                IEnumerable<int> distList = list.Distinct();  // .NET Framework version: >= 3.5
                Console.Write("distinct elements = ");
                foreach (int entry in distList)
                    Console.Write(entry + " ");
                Console.WriteLine();
            }

            {
                List<Pair> list = new List<Pair>();
                try
                {
                    list.Add(new Pair(0, 3));
                    list.Add(new Pair(1, 3));
                    list.Add(new Pair(2, 1));
                    list.Add(new Pair(3, 5));
                    list.Add(new Pair(4, 4));
                    list.Add(new Pair(5, 4));
                    list.Add(new Pair(6, 3));
                    list.Add(new Pair(7, 1));
                    list.Add(new Pair(8, 5));
                    list.Add(new Pair(9, 4));
                    list.Add(new Pair(10, 1));
					list.Add(new Pair(7, -1));
					list.Add(new Pair(8, -5));
					list.Add(new Pair(9, -4));
					list.Add(new Pair(10, -1));
				}
                catch (ArgumentException)
                {
                    Console.WriteLine("ArgumentException occurred !!!");
                }

                Console.WriteLine("list[0] = ({0}, {1})", list[0].key, list[0].val);

                list.Sort(new PairComparer());
                foreach (Pair pair in list)
                {
                    int key = pair.key;
                    int val = pair.val;
                    Console.WriteLine("Key: {0}, Value: {1}", key, val);
                }
            }
        }

		static void runHashSet()
		{
			{
				HashSet<int> set = new HashSet<int>();
				try
				{
					set.Add(3);
					set.Add(3);
					set.Add(1);
					set.Add(5);
					set.Add(4);
					set.Add(4);
					set.Add(3);
					set.Add(1);
					set.Add(5);
					set.Add(4);
					set.Add(1);
				}
				catch (ArgumentException)
				{
					Console.WriteLine("ArgumentException occurred !!!");
				}

				Console.Write("elements = ");
				foreach (int entry in set)
					Console.Write(entry + " ");
				Console.WriteLine();
			}

			{
				HashSet<Pair> set = new HashSet<Pair>(new PairEqualityComparer());
				try
				{
					set.Add(new Pair(0, 3));
					set.Add(new Pair(1, 3));
					set.Add(new Pair(2, 1));
					set.Add(new Pair(3, 5));
					set.Add(new Pair(4, 4));
					set.Add(new Pair(5, 4));
					set.Add(new Pair(6, 3));
					set.Add(new Pair(7, 1));
					set.Add(new Pair(8, 5));
					set.Add(new Pair(9, 4));
					set.Add(new Pair(10, 1));
					set.Add(new Pair(7, -1));
					set.Add(new Pair(8, -5));
					set.Add(new Pair(9, -4));
					set.Add(new Pair(10, -1));
				}
				catch (ArgumentException)
				{
					Console.WriteLine("ArgumentException occurred !!!");
				}

				foreach (Pair pair in set)
				{
					int key = pair.key;
					int val = pair.val;
					Console.WriteLine("Key: {0}, Value: {1}", key, val);
				}
			}
		}

        static void runDictionary()
        {
            Dictionary<int, int> dict = new Dictionary<int, int>();
            try
            {
                dict.Add(0, 3);
                dict.Add(1, 3);
                dict.Add(2, 1);
                dict.Add(3, 5);
                dict.Add(4, 4);
                dict.Add(5, 4);
                dict.Add(6, 3);
                dict.Add(7, 1);
                dict.Add(8, 5);
                dict.Add(9, 4);
                dict.Add(10, 1);
            }
            catch (ArgumentException)
            {
                Console.WriteLine("ArgumentException occurred !!!");
            }

            Dictionary<int, int>.KeyCollection keys = dict.Keys;
            Console.WriteLine("#keys = {0}", keys.Count);
            Dictionary<int, int>.ValueCollection vals = dict.Values;
            Console.WriteLine("#values = {0}", vals.Count);
        }

        static void runSortedList()
        {
            SortedList<int, int> slist = new SortedList<int, int>(new KeyComparer());
            try
            {
                slist.Add(0, 3);
                slist.Add(1, 3);
                slist.Add(2, 1);
                slist.Add(3, 5);
                slist.Add(4, 4);
                slist.Add(5, 4);
                slist.Add(6, 3);
                slist.Add(7, 1);
                slist.Add(8, 5);
                slist.Add(9, 4);
                slist.Add(10, 1);
            }
            catch (ArgumentException)
            {
                Console.WriteLine("ArgumentException occurred !!!");
            }

            try
            {
                foreach (KeyValuePair<int, int> pair in slist)
                {
                    int key = pair.Key;
                    int val = pair.Value;
                    Console.WriteLine("Key: {0}, Value: {1}", key, val);
                }
            }
            catch (KeyNotFoundException)
            {
                Console.WriteLine("KeyNotFoundException occurred !!!");
            }
        }
    }
}
