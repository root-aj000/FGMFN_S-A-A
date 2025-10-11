from diagrams import Diagram, Cluster, Edge
from diagrams.general.media import Television, Newspaper
from diagrams.onprem.client import User
from diagrams.aws.mobile import Mobile
from diagrams.aws.analytics import Quicksight, Kinesis, ElasticsearchService
from diagrams.aws.ml import Sagemaker, Rekognition, Comprehend, Forecast
from diagrams.aws.general import TraditionalServer
from diagrams.aws.compute import Compute
from diagrams.aws.engagement import Engage
from diagrams.aws.business import Business

with Diagram("Advertising Evolution Timeline: Metrics & Engagement Measurement", show=True, direction="TB"):
    
    with Cluster("Era 1: Traditional Print (1950s-1970s)", graph_attr={"bgcolor": "lightblue"}):
        print_media = Newspaper("Print Media")
        print_method = TraditionalServer("Mass Broadcasting")
        print_measure = Business("Measurement:\n• Circulation Numbers\n• Readership Surveys\n• Ad Space Revenue")
    
    with Cluster("Era 2: Television Broadcast (1980s-1990s)", graph_attr={"bgcolor": "lightgreen"}):
        tv_media = Television("TV Advertising")
        tv_method = TraditionalServer("Scheduled Broadcasts")
        tv_measure = Business("Measurement:\n• Nielsen Ratings\n• Gross Rating Points\n• Brand Awareness Studies")
    
    with Cluster("Era 3: Digital Online (2000s-2010s)", graph_attr={"bgcolor": "lightyellow"}):
        web_media = User("Online Advertising")
        web_method = Compute("Targeted Delivery")
        web_measure = ElasticsearchService("Measurement:\n• Click-Through Rates\n• Conversion Tracking\n• A/B Testing\n• Google Analytics")
    
    with Cluster("Era 4: Mobile & Social (2010s-2020s)", graph_attr={"bgcolor": "lightcoral"}):
        mobile_media = Mobile("Mobile & Social Ads")
        mobile_method = Engage("Interactive Engagement")
        mobile_measure = Kinesis("Measurement:\n• App Installs & Usage\n• Social Engagement\n• Location Data\n• Real-time Bidding")
    
    with Cluster("Era 5: AI Multimodal (2020s-Present)", graph_attr={"bgcolor": "lavender"}):
        ai_media = Sagemaker("AI-Powered Ads")
        ai_method = Forecast("Predictive Personalization")
        ai_measure = Quicksight("Measurement:\n• Sentiment Analysis\n• Cross-Modal Engagement\n• Trust Metrics\n• Predictive ROI\n• Real-time Optimization")
        ai_tech = [Rekognition("Computer Vision"), Comprehend("NLP"), Forecast("Predictive AI")]
    
    # Timeline flow
    print_media >> Edge(label="Broadcast Revolution", color="blue") >> tv_media
    tv_media >> Edge(label="Digital Transformation", color="green") >> web_media
    web_media >> Edge(label="Mobile Revolution", color="orange") >> mobile_media
    mobile_media >> Edge(label="AI Integration", color="red") >> ai_media
    
    # Method evolution
    print_method >> tv_method >> web_method >> mobile_method >> ai_method
    
    # Measurement evolution
    print_measure >> tv_measure >> web_measure >> mobile_measure >> ai_measure
    
    # AI technologies connection
    ai_media >> ai_tech