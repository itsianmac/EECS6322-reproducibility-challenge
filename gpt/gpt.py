import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from fake_useragent import UserAgent


class ChatGPTAutomation:

    def __init__(self, user_data_dir: str = './', wait_for_login: bool = False):
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
    def agent_turns(self):
        return self.driver.find_elements(by=By.CSS_SELECTOR,
                                         value='div.text-base > div.text-base > div.agent-turn')

    @property
    def response_numbers(self):
        return len(self.agent_turns)

    def ask(self, prompt: str):
        input_box = self.driver.find_element(by=By.XPATH, value='//textarea[contains(@id, "prompt-textarea")]')
        self.driver.execute_script(f"arguments[0].value = `{prompt}`;", input_box)
        current_response_numbers = self.response_numbers
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

    def regenerate(self):
        last_response = self.return_last_response()
        response_element = self.agent_turns[-1]
        response_element.find_elements(by=By.CSS_SELECTOR, value="span[data-state='closed']")[-2].click()

        def is_changed(_):
            return last_response != self.return_last_response()

        # wait for the new response to get started
        WebDriverWait(self.driver, 20).until(is_changed)
        # wait for the new response to be completed
        return self.wait_for_response()

    @staticmethod
    def is_long_response(response: str):
        if 'Copy code' in response:  # This means gpt is explaining the code
            return True
        return False

    def new_chat(self):
        # move to home page
        self.driver.get(r"https://chat.openai.com")

    def wait_for_response(self) -> str:
        """ Waits until the response from chatgpt is done """
        last_text = None

        def is_stable(_):
            nonlocal last_text
            response = self.return_last_response()
            try:
                return response == last_text or self.is_long_response(response)
            finally:
                last_text = response

        # wait until it's not changing anymore
        WebDriverWait(self.driver, 60, poll_frequency=1).until(is_stable)
        return self.return_last_response()

    def return_last_response(self):
        """ :return: the text of the last chatgpt response """
        response_elements = self.driver.find_elements(by=By.CSS_SELECTOR, value='div.text-message')
        return response_elements[-1].text

    @staticmethod
    def wait_for_human_verification():
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

    def quit(self):
        """ Closes the browser and terminates the WebDriver session."""
        print("Closing the browser...")
        self.driver.close()
        self.driver.quit()
