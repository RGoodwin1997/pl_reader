import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import pandas as pd
import base64
import io
# import tabula
import plreader as pl

app = dash.Dash()

# Callback to parse contents of a pdf
@app.callback(Output('pdf-viewer', 'data'),
              Output('pdf-viewer', 'columns'),
              Input('pdf-upload', 'contents'),
              State('pdf-upload', 'filename'),
              prevent_initial_call=True
              )
def pdf_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        #My pdf only has one page and one table with two columns
        print(filename)
        df = pl.main(io.BytesIO(decoded))
        # df.columns = ['Parameter', 'Value']
        
        return df.to_dict('records'), [{"name": i, "id": i, 'editable':True} for i in df.columns]

#Upload component:
pdf_load = dcc.Upload(id='pdf-upload',
                      children=html.Div(['Drag and Drop or ', html.A('Select PDF files')]),
                      style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
                             'borderWidth': '1px', 'borderStyle': 'dashed',
                             'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                      )

#Table to view output from pdf:
pdf_table = dash_table.DataTable(id='pdf-viewer',
                                 page_action='none',
                                 fixed_rows={'headers': True},
                                 style_table={'height': 500, 'overflowY': 'auto'},
                                 style_header={'overflowY': 'auto'},
                                 export_format="csv"
                                 )
#Place into the app
app.layout = html.Div([html.H4('Profit & Loss Statement Reader'),
                       pdf_load,
                       html.Br(),
                       pdf_table
                       ])


if __name__ == '__main__':
    app.run_server(debug = True)