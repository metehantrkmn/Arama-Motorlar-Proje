from seleniumwire import webdriver  # Import Selenium Wire
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime
from selenium.webdriver.common.action_chains import ActionChains


class Movie:
    def __init__(self, name, director, release_date, running_time, country, language, imdb_rating, genre, num_votes, plot):
        self.name = name  # Movie name
        self.director = director  # Director's name
        self.release_date = release_date  # Release date as a datetime object
        self.running_time = running_time  # Running time in minutes
        self.language = language  # Language of the movie
        self.imdb_rating = imdb_rating  # IMDB rating (float)
        self.genre = genre  # Genre(s) of the movie
        self.num_votes = num_votes  # Number of votes on IMDB
        self.plot = plot  # Plot description of the movie

    def to_dict(self):
        """Convert the movie object to a dictionary."""
        return {
            'name': self.name,
            'director': self.director,
            'release_date': self.release_date.strftime('%Y-%m-%d') if isinstance(self.release_date, datetime.date) else self.release_date,
            'running_time': self.running_time,
            'language': self.language,
            'imdb_rating': self.imdb_rating,
            'genre': self.genre,
            'num_votes': self.num_votes,
            'plot': self.plot
        }

    def __str__(self):
        """String representation of the Movie object."""
        return f"Movie: {self.name} ({self.release_date.year if isinstance(self.release_date, datetime.date) else 'Unknown'}), Directed by: {self.director}"


# Set up Selenium Wire options for custom headers
seleniumwire_options = {
    'request_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
}

# Initialize WebDriver with Selenium Wire options
driver = webdriver.Chrome(seleniumwire_options=seleniumwire_options)

# Open IMDb Top 250 page
url = 'https://www.imdb.com/chart/top/?ref_=nv_mv_250_6'
driver.get(url)

# Wait for the page to load
time.sleep(5)

# Locate movie rows
rows = driver.find_elements(By.CLASS_NAME, "ipc-metadata-list-summary-item")
print(f"Found {len(rows)} movies.")

# Iterate over each row to scrape data
movies = []

for row in rows:
    try:
        # Extract movie name
        name = row.find_element(By.CLASS_NAME, "ipc-title__text").text

        # Extract additional details from the row
        # this line extracts a div includes spans of date and running time and pg-13
        div_spans = row.find_element(By.CLASS_NAME, "sc-300a8231-6")
        spans = div_spans.find_elements(By.CLASS_NAME, "sc-300a8231-7")
        print("deneme ilk: " + spans[0].text)
        print("deneme ilk: " + spans[1].text)
        other_spans = row.find_elements(By.CLASS_NAME, "ipc-rating-star span")
        print("deneme iki: " + other_spans[0].text)
        print("deneme iki: " + other_spans[1].text)

        # Click the button to open the modal
        button = row.find_element(By.CLASS_NAME, "ipc-icon-button")
        print(button.tag_name)
        #ActionChains(driver).move_to_element(button).click().perform()
        driver.execute_script("arguments[0].click();", button)

        info_section = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'ipc-promptable-base'))
        )

        #print(info_section)

        # Extract details from the modal
        #info_section = driver.find_element(By.CLASS_NAME, "ipc-promptable-base")
        print("deneme uc " + info_section.tag_name)


        time.sleep(3)

        panel = info_section.find_element(By.CLASS_NAME, "ipc-promptable-base__panel")
        print("hata deneme 1")
        outer_div = panel.find_element(By.CLASS_NAME, "sc-b90eafb6-2")
        print("deneme hata 2")
        # gets 2 ul from modal, first includes rating etc and second includes GENRE
        ul = outer_div.find_elements(By.CLASS_NAME, "ipc-inline-list")
        print(len(ul))
        genre = ul[1].find_element(By.TAG_NAME, "li").text
        print(genre)

        plot = info_section.find_element(By.CLASS_NAME, "sc-8407191a-2").text
        print(plot)


        director = info_section.find_element(By.CLASS_NAME, "sc-1582ce06-2").find_element(By.TAG_NAME, "a").text

        release_date = spans[0].text 

        # Extract running time
        running_time = spans[1].text if len(spans) > 1 else None

        # Extract rating and vote count
        imdb_rating = other_spans[0].text if other_spans else None
        num_votes = other_spans[1].text if len(other_spans) > 1 else None

        # Close the modal after scraping
        close_button = info_section.find_element(By.CLASS_NAME, "ipc-promptable-base__panel").find_element(By.CLASS_NAME, "ipc-icon-button")
        driver.execute_script("arguments[0].click();", close_button)
        time.sleep(3)

        # Append movie details to the list
        movie_details = Movie(
            name=name,
            release_date=release_date,
            running_time=running_time,
            country=None,  # Country is not explicitly available on the page
            language=None,  # Language is not explicitly available on the page
            imdb_rating=imdb_rating,
            genre=genre,
            num_votes=num_votes,
            plot=plot,
            director=director
        )

        # Print movie details
        print(movie_details.to_dict())
        movies.append(movie_details)

    except Exception as e:
        print(f"Error processing movie: {e}")
        continue

# Close the browser
driver.quit()

# Print all collected movies
print(f"\nScraped {len(movies)} movies:")
for movie in movies:
    print(movie.to_dict())
