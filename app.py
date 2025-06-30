# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
    "The Evolution of Sports Training and Performance: Over the last few decades, sports training has evolved dramatically due to advancements in technology, nutrition, and science. Traditional training focused largely on physical conditioning, but modern athletes now benefit from data-driven approaches. For example, wearable tech like WHOOP and Garmin devices track recovery, heart rate variability, and sleep patterns. Nutritionists tailor diets to optimize muscle repair and energy efficiency. Even mental training is emphasized; elite teams like the Seattle Seahawks hire sports psychologists to improve focus and resilience. These developments have not only raised individual performance but have also reduced injury rates. Critics argue that this reliance on tech may lead to over-monitoring and diminished athlete intuition, but evidence shows measurable performance gains. The evolution reflects a broader shift: sports today are as much about strategy and precision as strength and endurance.",

    "The Role of Sports in Shaping National Identity: Sports have long played a central role in forging national identity. International events like the Olympics and the FIFA World Cup often serve as platforms for countries to assert pride and unity. In 1995, South Africas rugby team famously symbolized reconciliation after apartheid, a moment captured in the film Invictus. Similarly, the U.S. “Miracle on Ice” during the 1980 Winter Olympics wasnt just a hockey victory—it was a Cold War-era morale boost. However, the power of sports nationalism has a double edge. While it unites people, it can also intensify rivalries and politicize athletes. Think of the debates surrounding players kneeling during anthems to protest injustice. Is nationalism in sports a unifying celebration or a distraction from deeper issues? The answer may depend on how responsibly it's wielded. Sports are more than games; theyre symbols—flexible, potent, and capable of shaping how people see their country and themselves.",

    "The Impact of Youth Sports on Child Development: Participation in youth sports has a profound impact on a childs development. According to the Aspen Institutes Project Play, kids involved in regular sports are more likely to maintain healthy lifestyles and develop critical life skills such as teamwork, discipline, and resilience. For example, team sports teach cooperation and shared responsibility, while individual sports like gymnastics foster goal-setting and self-discipline. However, early specialization in one sport—a growing trend—raises concerns. Studies show it can lead to burnout, overuse injuries, and psychological stress. Moreover, the pressure from parents and coaches sometimes turns play into performance, stripping the joy from the experience. A balanced approach, emphasizing fun, variety, and long-term development, appears to be the healthiest route. Youth sports should be less about trophies and more about teaching habits and attitudes that last a lifetime.",

    "Economic Influence of Major Sporting Events: Major sporting events like the Super Bowl, World Cup, and Olympics generate billions in revenue, but the economic impact is more complex than headlines suggest. Host cities often spend heavily on infrastructure, such as Brazils $13 billion investment for the 2014 World Cup. While these events can boost tourism and create jobs, the benefits are uneven. Many stadiums become “white elephants,” underused after the spectacle ends. Studies from Oxford and the University of Zurich show that projected economic gains are frequently overstated. Still, there are exceptions: Barcelonas 1992 Olympics revitalized its urban image and economy. The key question is whether short-term spectacle justifies long-term expense. Should public funds be diverted to sports, or would education, health, and housing be a better investment? Evaluating economic impact requires more than tallying receipts—it demands scrutiny of opportunity cost, sustainability, and equity.",

    "The Future of Sports in a Digitally Connected World: The future of sports is being shaped by digital innovation. From VR training environments to blockchain-based fan engagement, technology is redefining how games are played and experienced. Athletes are using AI-powered analytics to tailor their techniques, while fans interact with sports through fantasy leagues, Twitch streams, and virtual stadiums. Companies like Sorare and NBA Top Shot are experimenting with digital collectibles, creating new revenue streams and communities. However, this transformation raises concerns: Does increasing reliance on screens and algorithms risk diluting the physical essence of sports? Are we prioritizing entertainment over authenticity? The metaverse may soon host virtual Olympics, blurring the line between athlete and avatar. This could democratize participation—or commodify it further. As sports evolve digitally, the tension between innovation and tradition will shape what we define as the essence of the game."
    ]

    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection



def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 0.8:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
st.title("🏟️⚽ Game On: Exploring the Power, Progress, and Future of Sports 🏀🏆")
st.markdown("### 🏟️ Welcome to **Sports Central!**")
st.markdown("*Your interactive sports knowledge assistant*")


# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.write("📚 Have a question about sports? 🤔 Dive into the topics below and ask anything about training, youth development, national identity, mega-events, or the digital future of sports!")

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("🤔 What’s on your mind about sports? Ask away!")

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("🏁 Ready, Set, Answer!", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("🏃‍♂️ Analyzing game data..."):
            answer = get_answer(collection, question)
        
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**📣 Answer:**")
        st.write(answer)

        st.success("🎯 Found a solid sports answer for you!")
        st.info("📣 Tip: Try asking about a specific sport, trend, or industry shift.")
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("⚠️ Please enter a question!")

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander("📘 How to Use This Sports Q&A App"):
    st.write("""
    1. I created this system with knowledge about:
       - Athlete training
       - Youth sports
       - National identity in sports
       - Sports economics
       - Digital innovation in sports
    2. Type a question based on any of the topics.
    3. Click '🏁 Ready, Set, Answer!' (or your chosen button).
    4. Review the system’s response and explore further!

    Example questions you can ask:
    – How has technology changed athlete performance?
    – What role does youth sport play in education?
    – Are major sporting events economically beneficial?
    – How is social media influencing the sports industry?
    """)



# TO RUN: Save as app.py, then type: streamlit run app.py

"""
STREAMLIT BUILDING BLOCKS SUMMARY:
================================

1. st.title(text) 
   - Creates the main heading of your app
   - Appears as large, bold text at the top

2. st.write(text)
   - Displays text, data, or markdown content
   - Most versatile output function in Streamlit
   - Can display simple text, formatted text, or data

3. st.text_input(label, placeholder="hint")
   - Creates a text box where users can type
   - Returns whatever the user types
   - Label appears above the box

4. st.button(text, type="primary")
   - Creates a clickable button
   - Returns True when clicked, False otherwise
   - Use in 'if' statements to trigger actions
   - type="primary" makes it blue and prominent

5. st.spinner(text)
   - Shows a spinning animation with custom text
   - Use with 'with' statement for code that takes time
   - Automatically disappears when code finishes

6. st.expander(title)
   - Creates a collapsible section
   - Users can click to expand/collapse content
   - Great for help text or optional information
   - Use with 'with' statement for content inside

HOW THE APP FLOW WORKS:
======================

1. User opens browser → Streamlit loads the app
2. setup_documents() runs → Creates document database
3. st.title() and st.write() → Display app header
4. st.text_input() → Shows input box for questions  
5. st.button() → Shows the "Get Answer" button
6. User types question and clicks button:
   - if statement triggers
   - st.spinner() shows loading animation
   - get_answer() function runs in background
   - st.write() displays the result
7. st.expander() → Shows help section at bottom

WHAT HAPPENS WHEN USER INTERACTS:
=================================

- Type in text box → question variable updates automatically
- Click button → if st.button() becomes True
- Spinner shows → get_answer() function runs
- Answer appears → st.write() displays the result
- Click expander → help section shows/hides

This creates a simple but complete web application!
"""