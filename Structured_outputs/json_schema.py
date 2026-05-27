from typing import Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

json_schema={
    'title':'Review Assistant',
    'type':'object',
    'properties':{
        'summary':{
            'type':'string',
            'description':"I want you to summarize the entire Review"
        },
        'sentiment':{
            'type':'string',
            'enum':['Positive','Negative','Neutral'],
            'description':"The sentiment can either be Positive , Negative or Neutral"
        },
        'points':{
            'type':'string',
            'items':{
                'type':'string'
            },
            'description':"Give me a list of major points discussed inside this review"
        },
        'pros':{
            'type':["array"],
            'items':{
                'type':'string'
            },
            'description':"List of all the pros"
        },

        'cons':{
        'type':["array"],
        'items':{
            'type':'string'
        },
        'description':"List of all the cons"
    }

    }
}


structured_op=model.with_structured_output(json_schema)

query="""’m a freelance photographer using this mostly for Lightroom Classic, Photoshop, and occasional 4K video work. I upgraded from the 14” M1 MacBook Pro (2021).

Positives:
• M4 is noticeably faster for AI masking, batch exports, and working with large RAW files
• Totally silent – no fan noise at all
• Better battery life (about 1–1.5 hours longer in real-world use)
• Slightly brighter display, improved contrast in HDR content
• WiFi 6E makes file transfers snappier (if your router supports it)

Neutral:
• Same excellent build quality and keyboard
• Port layout unchanged – still very practical

Negatives:
• Performance jump over M1 is good but not massive — mostly noticeable in AI-heavy workflows
• Price is steep if your M1 is still running fine

Bottom Line:
If you’re coming from an Intel Mac or your M1 is starting to show its age, the M4 14” is a solid upgrade: faster, cooler, and better battery life. But if your M1 is still handling your workload fine, you won’t see a night-and-day difference. """

result=structured_op.invoke(query)

print(result)