import time
from typing import List

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from fake_useragent import UserAgent


class GPTClient:
    """ A class to automate the chat with chatgpt using selenium

    Args:
        user_data_dir (str): the path to the user data directory of the browser. Defaults to './user-data'
        wait_for_login (bool): whether to wait for the user to log in or complete the human verification process.
            You will only need to set this to True if you are using a new user data directory. Defaults to False.
    """

    def __init__(self, user_data_dir: str = './user-data', wait_for_login: bool = False):
        op = webdriver.ChromeOptions()
        op.add_argument(f"user-agent={UserAgent.random}")

        self.driver = uc.Chrome(options=op, user_data_dir=user_data_dir, use_subprocess=True, port=34562)
        url = r"https://chat.openai.com"
        self.driver.get(url)
        if wait_for_login:
            self.wait_for_human_verification()
        # wait for the prompt box to appear
        WebDriverWait(self.driver, 60).until(
            EC.presence_of_element_located((By.XPATH, '//textarea[contains(@id, "prompt-textarea")]')))

    @property
    def agent_turns(self) -> List[WebElement]:
        """ the chatgpt response containers

        Returns:
            list: the chatgpt response containers
        """
        return self.driver.find_elements(by=By.CSS_SELECTOR,
                                         value='div.text-base > div.text-base > div.agent-turn')

    @property
    def response_numbers(self) -> int:
        """ the number of chatgpt responses

        Returns:
            int: the number of chatgpt responses
        """
        return len(self.agent_turns)

    def ask(self, prompt: str) -> str:
        """ Sends a prompt to chatgpt and returns the response. If the response is empty or too long,
        it will regenerate it.
        
        Args:
            prompt (str): the prompt to send to chatgpt
            
        Returns:
            str: the response from chatgpt
        """
        # fill the prompt box
        input_box = self.driver.find_element(by=By.XPATH, value='//textarea[contains(@id, "prompt-textarea")]')
        self.driver.execute_script(f"arguments[0].value = `{prompt}`;", input_box)
        current_response_numbers = self.response_numbers

        # send the prompt
        input_box.send_keys(Keys.RETURN)
        input_box.submit()

        # wait for the response to get started
        WebDriverWait(self.driver, 20).until(lambda _: self.response_numbers > current_response_numbers)

        # wait for the response to be completed
        response = self.wait_for_response()

        # if the response is empty or too long, regenerate it
        while response.strip() == '' or self.is_long_response(response):
            response = self.regenerate()
        return response

    def regenerate(self) -> str:
        """ Regenerates the last response from chatgpt

        Returns:
            str: the new response from chatgpt
        """
        previous_response = self.last_response

        # click the regenerate button
        response_element = self.agent_turns[-1]
        response_element.find_elements(by=By.CSS_SELECTOR, value="span[data-state='closed']")[-2].click()

        def is_changed(_):
            return previous_response != self.last_response

        # wait for the new response to get started
        WebDriverWait(self.driver, 20).until(is_changed)

        # wait for the new response to be completed
        return self.wait_for_response()

    @staticmethod
    def is_long_response(response: str) -> bool:
        """ Checks if the response is too long

        Args:
            response (str): the response to check

        Returns:
            bool: True if the response is too long, False otherwise
        """
        if 'Copy code' in response:  # This means gpt is explaining the code
            return True
        return False

    def new_chat(self) -> None:
        """ Opens a new chat with chatgpt """
        self.driver.get(r"https://chat.openai.com")

    def wait_for_response(self) -> str:
        """ Waits until the response from chatgpt is completed and returns it

        Returns:
            str: the response from chatgpt
        """
        last_text = None

        def is_stable(_):
            nonlocal last_text
            response = self.last_response
            try:
                # if it's the same as the last one, it's stable
                # or if it's already too long, we'd rather not wait for it continue
                return response == last_text or self.is_long_response(response)
            finally:
                last_text = response

        # wait until it's not changing anymore
        WebDriverWait(self.driver, 60, poll_frequency=1).until(is_stable)
        return self.last_response

    @property
    def last_response(self) -> str:
        """ The last response from chatgpt

        Returns:
            str: the last response from chatgpt
        """
        response_elements = self.driver.find_elements(by=By.CSS_SELECTOR, value='div.text-message')
        return response_elements[-1].text

    @staticmethod
    def wait_for_human_verification() -> None:
        """ Waits for the user to complete the log-in or the human verification process. """
        print("You need to manually complete the log-in or the human verification if required.")

        while True:
            user_input = input(
                "Enter 'y' if you have completed the log-in or the human verification, or 'n' to check again: ").lower()

            if user_input == 'y':
                print("Continuing with the automation process...")
                break
            elif user_input == 'n':
                print("Waiting for you to complete the human verification...")
                time.sleep(5)  # You can adjust the waiting time as needed
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    def quit(self) -> None:
        """ Closes the browser and terminates the WebDriver session."""
        print("Closing the browser...")
        self.driver.close()
        self.driver.quit()
