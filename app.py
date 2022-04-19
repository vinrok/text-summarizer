import streamlit as st
from transformers import pipeline
import gradio as gr
from gradio.mix import Parallel, Series

@st.cache(allow_output_mutation=True)
def summarize_model():
    model = pipeline("summarization")
    return model

summ = summarize_model()
st.title("Text Summarizer")
st.subheader("Paste any article in the text area below and get the summary by clicking on 'Summarize Text' button")
st.caption("Text summarization using HuggingFace's transformers pre-trained model")

sentence = st.text_area('Paste your copied data here...', height=100)
button = st.button("Summarize Text")

max_lengthy = st.sidebar.slider('Maximum summary length (words)', min_value=30, max_value=700, value=100, step=10)
num_beamer = st.sidebar.slider('Speed vs quality of Summary (1 is fastest but less accurate)', min_value=1, max_value=8, value=4, step=1)

with st.spinner("Summarizing..."):
    if button and sentence:
        summary = summ(sentence, max_length = max_lengthy, min_length = 50, num_beams=num_beamer, do_sample=True,early_stopping=True, repetition_penalty=1.5, length_penalty=1.5)[0]
	st.write(summary['summary_text'])


# io1 = gr.Interface.load('huggingface/facebook/bart-large-cnn')
# desc = "Text summarization using HuggingFace's transformers pre-trained model"

# w = """Daniel Buren has created a large expensive Painting on Black self-adhesive vinyl under Plexiglas on black wall in the classic year of 1989.0 Using four small squares to form a large square the artist plays here with metamorphosis and geometric combination and puts at the same level content and container he confuses framing and framed he makes the contingent element an essential element Another principle used by Daniel Buren was the fragmenting of a square form using this means to generate other elementary forms Once installed on the walls the latter restores the original quadrilateral but developed in space The interdependence between the varying place of exhibition and the components is the foundation of Buren s thought as an artist the artwork is not an object it is a place from which to perceive the world 
# """

# x = """Juan UslÃ© has created a large expensive Painting on  vinyl, dispersion and dry pigment on canvas in the classic year of 2018.0 Juan Usl paintings are complex interactions and incorporate a great diversity of art historical references sensory and mental impressions various pictorial languages the gesture of painting and how the matter paints itself functions The first step is the canvas s preparation with multiple layers of gesso which is a key element that will remain visible This continuous manifestation of the gesso also conveys a philosophical resonance for Usl namely that the beginning is present at the end that the painting is a self contained entity a complete object not merely within its four sides but in the vertical layering of its surface as well The process of painting consists of a natural harmony between the manual act and the intellectual decisions Movement or even better displacement is a thematic key in his work In Usl s work we can read a constant dichotomy between opposing and complementary elements at the same time order and chaos presence and absence flatness and depth Most of his paintings are a juxtaposition of color areas and lines structures that seem to come and go like the fragments of a story The Artist presents his work as a temporary delimitation of infinite surfaces or as fragments from an infinite structure of lines Particularly well known is the series of paintings called So que Revelabas Dream that revelead In these works the artist through a deeply introspective practice seems to give shape to his more intimate self While painting Usl tries to connect rhythmically with his palpitation making each stroke a symbolic representation of the beating of his heart return the theme of disorientation no longer understood only in a physical sense but also and above all in a more temporal and perceptive way His paintings take the viewer into a labyrinthine space in which the articulation seems to indicate a specific direction while paradoxically it leaves open the way to interpretation 
# """

# y = """Marisa Merz has created a medium expensive Installation on Mixed media on plywood, copper in the classic year of 2010.0 Marisa Merz 1925 2019 Italy was one of the key personalities and the only woman associated with the Arte Povera movement in the late 60s and 70s Known for the unusual use of materials such as copper wire clay and wax the sculptures and drawings of Merz reflect her poetic sensibility and vision for art and life By employing abstract organic figures Merz creates familiar portraits and sculptures which insist on subjectiveness while emitting both a constantly changing message as well as the artist s belief that every shape is fluid and should be able to transform into any other shape Merz does not distinguish between art and life employing handcraft and unconventional materials in order to explore the infinite possibilities of everyday life while in many cases she adopts traditional techniques associated with the female domestic environment such as knitting The idea of home acting as a sphere engulfing privacy familiarity and female concepts is one of the central references to her work 
# """

# z = """Judith Eisler has created a small nominal Painting on Oil on canvas in the modern year of 2020.0 Judith Eisler b 1962 Newark NJ presents Faye a new oil painting depicting the actress Faye Dunaway adapted from the widely acclaimed satirical motion picture Network 1976 written by Paddy Chayefsky and directed by Sidney Lumet Eisler has recently expanded her source material to include contemporary imagery appropriated from politics sports and entertainment however in Faye she returns to her longstanding practice of painting cinematic close ups taken from stills from classic 1960s and 1970s films In Network Dunaway portrays Diana Christensen a producer for the fictional television network UBS who devises increasingly dramatic performances for a veteran news anchorman Howard Beale played by Peter Finch in an effort to increase ratings Eisler s portrait is tightly cropped zoomed in on Dunaway s face and framed by coiffed brunette curls The artist playfully draws connections between the film s title and the mediums of television screens and social media Eisler skillfully incorporates the pixelated striations that often occur when photographing TV monitors revealing the multi step process involved in the creation of painting film stills These striations are a result of a moir pattern a large scale interference pattern that occurs when photographing a television screen a failure of video pixels to coordinate with the screen s display These distorted bands of red and blue light sweep across Dunaway s visage mapping her intensely focused gaze 
# """

# sample = [[w],[x],[y],[z]]

# iface = Parallel(io1, 
#                  theme='peach', 
#                  title= 'Hugging Face Text Summarizer', 
#                  description = desc,
#                  examples=sample, #replace "sample" with directory to let gradio scan through those files and give you the text
#                  inputs = gr.inputs.Textbox(lines = 10, label="Text"),
#                  outputs = "text")
	
# iface.launch(inline = False)
