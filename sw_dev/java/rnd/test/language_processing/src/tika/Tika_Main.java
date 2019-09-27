package tika;

import java.io.IOException;
import java.io.InputStream;

import org.apache.tika.Tika;
import org.apache.tika.detect.Detector;
import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.mime.MediaType;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.sax.BodyContentHandler;

public class Tika_Main {

	public static void run(String[] args)
	{
		// REF [site] >> https://www.baeldung.com/apache-tika
		whenUsingDetector_thenDocumentTypeIsReturned();
		whenUsingParser_thenContentIsReturned();
		whenUsingParser_thenMetadataIsReturned();
	}

	static String detectDocTypeUsingDetector(InputStream stream) throws IOException {
		Detector detector = new Detector();
		Metadata metadata = new Metadata();

		MediaType mediaType = detector.detect(stream, metadata);
		return mediaType.toString();
	}

	static String detectDocTypeUsingFacade(InputStream stream) throws IOException {
		Tika tika = new Tika();
		String mediaType = tika.detect(stream);
		return mediaType;
	}

	static String extractContentUsingParser(InputStream stream) throws IOException, TikaException, SAXException {
		Parser parser = new AutoDetectParser();
		ContentHandler handler = new BodyContentHandler();
		Metadata metadata = new Metadata();
		ParseContext context = new ParseContext();

		parser.parse(stream, handler, metadata, context);
		return handler.toString();
	}

	static String extractContentUsingFacade(InputStream stream) throws IOException, TikaException {
		Tika tika = new Tika();
		String content = tika.parseToString(stream);
		return content;
	}

	static Metadata extractMetadatatUsingParser(InputStream stream) throws IOException, SAXException, TikaException {
		Parser parser = new AutoDetectParser();
		ContentHandler handler = new BodyContentHandler();
		Metadata metadata = new Metadata();
		ParseContext context = new ParseContext();

		parser.parse(stream, handler, metadata, context);
		return metadata;
	}

	static Metadata extractMetadatatUsingFacade(InputStream stream) throws IOException, TikaException {
		Tika tika = new Tika();
		Metadata metadata = new Metadata();

		tika.parse(stream, metadata);
		return metadata;
	}

	@Test
	void whenUsingDetector_thenDocumentTypeIsReturned() throws IOException
	{
		InputStream stream = this.getClass().getClassLoader().getResourceAsStream("tika.txt");
		String mediaType = Tika_Main.detectDocTypeUsingDetector(stream);

		assertEquals("application/pdf", mediaType);

		stream.close();
	}

	@Test
	void whenUsingParser_thenContentIsReturned() throws IOException, TikaException, SAXException
	{
		InputStream stream = this.getClass().getClassLoader().getResourceAsStream("tika.docx");
		String content = Tika_Main.extractContentUsingParser(stream);

		assertThat(content, containsString("Apache Tika - a content analysis toolkit"));
		assertThat(content, containsString("detects and extracts metadata and text"));

		stream.close();
	}

	@Test
	void whenUsingParser_thenMetadataIsReturned() throws IOException, TikaException, SAXException
	{
		InputStream stream = this.getClass().getClassLoader().getResourceAsStream("tika.xlsx");
		Metadata metadata = Tika_Main.extractMetadatatUsingParser(stream);

		assertEquals("org.apache.tika.parser.DefaultParser", metadata.get("X-Parsed-By"));
		assertEquals("Microsoft Office User", metadata.get("Author"));

		stream.close();
	}
}
