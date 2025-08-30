import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

DEFAULT_SPREADSHEET = os.getenv('GOOGLE_SPREADSHEET_NAME', 'AlphaBot Match Data v1')

# Only support docker secret path(s); keep backwards-compatible .json filename if used
DEFAULT_CREDS_PATH = (
    '/run/secrets/google_api_creds'
    if os.path.exists('/run/secrets/google_api_creds')
    else '/run/secrets/google_api_creds.json'
)


DEFAULT_SHEET = 'RAW'

class GoogleSheetsManager:
    def __init__(self, creds_file=DEFAULT_CREDS_PATH, spreadsheet_name=DEFAULT_SPREADSHEET):
        # Use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if not creds_file or not os.path.exists(creds_file):
            raise FileNotFoundError(f"Google credentials file not found: {creds_file}. Mount docker secret 'google_api_creds' to /run/secrets/google_api_creds (or .json).")
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
        self.client = gspread.authorize(creds)

        # Open the spreadsheet
        self.spreadsheet = self.client.open(spreadsheet_name)

    def upload_df(self, df, sheet_name=DEFAULT_SHEET, mode='append'):
        # Check if the sheet exists, if not create it
        try:
            sheet = self.spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            self.create_sheet(sheet_name)
            sheet = self.spreadsheet.worksheet(sheet_name)

        # Convert DataFrame to list of lists
        data = df.values.tolist()
        headers = df.columns.tolist()

        if mode == 'append':
            # Append the data
            sheet.append_rows(data)
        elif mode == 'replace':
            # Clear the sheet and set new headers and data
            sheet.clear()
            sheet.append_row(headers)
            sheet.append_rows(data)
        else:
            raise ValueError("Mode not recognized. Use 'append' or 'replace'.")

    def create_sheet(self, sheet_name):
        # Add a new worksheet with the given name
        self.spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
