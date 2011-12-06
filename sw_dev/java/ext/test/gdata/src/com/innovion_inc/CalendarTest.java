package com.innovion_inc;

import com.google.gdata.client.calendar.CalendarService;
import com.google.gdata.data.PlainTextConstruct;
import com.google.gdata.data.calendar.CalendarEntry;
import com.google.gdata.data.calendar.CalendarEventEntry;
import com.google.gdata.data.calendar.CalendarFeed;
import com.google.gdata.data.calendar.ColorProperty;
import com.google.gdata.data.calendar.HiddenProperty;
import com.google.gdata.data.calendar.SelectedProperty;
import com.google.gdata.data.calendar.TimeZoneProperty;
import com.google.gdata.data.extensions.Where;
import com.google.gdata.util.AuthenticationException;
import com.google.gdata.util.ServiceException;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

public class CalendarTest {
	public static void runAll()
	{
		CalendarService calendarService = new CalendarService("Innvion-TimeTracker-1.0");

		// authentication
		try
		{
			calendarService.setUserCredentials(userName, userPassword);
		}
		catch (final AuthenticationException e)
		{
			e.printStackTrace();
		}

		//
		System.out.println("----------* list all the calendars");
		listCalendars(calendarService);

		// calendar
		System.out.println("\n----------* add a new calendar");
		CalendarEntry calendarEntry = addCalendar(calendarService);
		System.out.println("\n----------* update a calendar");
		CalendarEntry updatedCalendarEntry = updateCalendar(calendarEntry);
		System.out.println("\n----------* remove a new calendar");
		removeCalendar(updatedCalendarEntry);
		
		// calendar subscription
		System.out.println("\n----------* add a new calendar subscription");
		CalendarEntry newSubscription = addCalendarSubscription(calendarService);
		System.out.println("\n----------* update a calendar subscription");
	    CalendarEntry updatedSubscription = updateCalendarSubscription(newSubscription);
		System.out.println("\n----------* remove a calendar subscription");
	    removeCalendarSubscription(updatedSubscription);
	}

	// list calendars
	private static void listCalendars(CalendarService calendarService)
	{
		try
		{
			URL feedUrl = allCalendarsFeedUrl;
	        //URL feedUrl = primaryCalendarFeedUrl;

			CalendarFeed resultFeed = calendarService.getFeed(feedUrl, CalendarFeed.class);

			System.out.println("My calendars:");
	        for (CalendarEntry entry : resultFeed.getEntries())
	        {
	        	System.out.println("\t" + entry.getTitle().getPlainText());
	        }
		}
		catch (final MalformedURLException e)
		{
			e.printStackTrace();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
	}
		
	// add a calendar entry
	private static CalendarEntry addCalendar(CalendarService calendarService)
	{
		try
		{
		    CalendarEntry calendarEntry = new CalendarEntry();

		    calendarEntry.setTitle(new PlainTextConstruct("Little League Schedule"));
		    calendarEntry.setSummary(new PlainTextConstruct("This calendar contains the practice schedule and game times."));
		    calendarEntry.setTimeZone(new TimeZoneProperty("America/Los_Angeles"));

		    calendarEntry.setHidden(HiddenProperty.FALSE);
		    calendarEntry.setColor(new ColorProperty(RED));
		    calendarEntry.addLocation(new Where("", "", "Oakland"));

	        // insert the calendar
	        //URL postURL = primaryCalendarFeedUrl;
		    URL postURL = ownCalendarsFeedUrl;
		    CalendarEntry insertedEntry = calendarService.insert(postURL, calendarEntry);
	        if (null == insertedEntry)
	        {
	        	System.out.println("error: fail to add calendar entry");
	        }
	        
	        return insertedEntry;
		}
		catch (final MalformedURLException e)
		{
			e.printStackTrace();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
		
		return null;
	}
	
	// update a calendar entry
	private static CalendarEntry updateCalendar(CalendarEntry calendarEntry)
	{
		try
		{
			calendarEntry.setTitle(new PlainTextConstruct("New title"));
			calendarEntry.setColor(new ColorProperty(GREEN));
			calendarEntry.setSelected(SelectedProperty.TRUE);
			
			return calendarEntry.update();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
		
		return null;
	}
	
	// remove a calendar entry
	private static void removeCalendar(CalendarEntry calendarEntry)
	{
		try
		{
			calendarEntry.delete();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
	}
	
	// add a calendar entry
	private static CalendarEntry addCalendarSubscription(CalendarService calendarService)
	{
		try
		{
		    CalendarEntry calendar = new CalendarEntry();
		    calendar.setId(DOODLES_CALENDAR_ID);

		    URL postURL = allCalendarsFeedUrl;
		    return calendarService.insert(postURL, calendar);
		}
		catch (final MalformedURLException e)
		{
			e.printStackTrace();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
		
		return null;
	}
	
	// update a calendar entry
	private static CalendarEntry updateCalendarSubscription(CalendarEntry calendarEntry)
	{
		try
		{
			calendarEntry.setColor(new ColorProperty(RED));

		    return calendarEntry.update();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
		
		return null;
	}
	
	// remove a calendar entry
	private static void removeCalendarSubscription(CalendarEntry calendarEntry)
	{
		try
		{
			calendarEntry.delete();
		}
		catch (final ServiceException e)
		{
			e.printStackTrace();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
	}

	// the base URL for a user's calendar metafeed (needs a username appended).
	private static final String METAFEED_URL_BASE = "https://www.google.com/calendar/feeds/";
	// the string to add to the user's metafeedUrl to access the allcalendars feed.
	private static final String ALLCALENDARS_FEED_URL_SUFFIX = "/allcalendars/full";
	// the string to add to the user's metafeedUrl to access the owncalendars feed.
	private static final String OWNCALENDARS_FEED_URL_SUFFIX = "/owncalendars/full";
	// the string to add to the user's metafeedUrl to access the event feed.
	private static final String EVENT_FEED_URL_SUFFIX = "/private/full";
	
	private static final String userName = "innovion.inc@gmail.com";
	private static final String userPassword = "innovion2gmail";

	// the URL for the allcalendars feed of the specified user.
	// (e.g. https://www.google.com/calendar/feeds/jdoe@gmail.com/allcalendars/full)
	// The allcalendars feed is a private read/write feed that is used for managing subscriptions and personalization settings of a user's calendars.
	// Shows only public events.
	// Always read-only.
	private static URL allCalendarsFeedUrl = null;
	// the URL for the owncalendars feed of the specified user.
	// (e.g. https://www.google.com/calendar/feeds/jdoe@gmail.com/owncalendars/full)
	// the owncalendars feeds is a private read/write feed that can be used to manage calendars that a user owns.
	// Shows both public and private events.
	// Potentially read/write.
	private static URL ownCalendarsFeedUrl = null;

    // the calendar ID of the public Google Doodles calendar
    private static final String DOODLES_CALENDAR_ID = "c4o4i7m2lbamc4k26sc2vokh5g%40group.calendar.google.com";

    private static final String RED = "#A32929";
	private static final String BLUE = "#2952A3";
	private static final String GREEN = "#0D7813";
	
	static
	{
		try
		{
			allCalendarsFeedUrl = new URL(METAFEED_URL_BASE + userName + ALLCALENDARS_FEED_URL_SUFFIX);
			//allCalendarsFeedUrl = new URL("https://www.google.com/calendar/feeds/default/allcalendars/full");
			
			ownCalendarsFeedUrl = new URL(METAFEED_URL_BASE + userName + OWNCALENDARS_FEED_URL_SUFFIX);
		}
		catch (final MalformedURLException e)
		{
			e.printStackTrace();
		}
	}
}
