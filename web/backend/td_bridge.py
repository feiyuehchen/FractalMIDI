# TouchDesigner Bridge
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class TouchDesignerBridge:
    def __init__(self):
        self.clients = set()
        
    async def connect(self, websocket):
        self.clients.add(websocket)
        logger.info(f"TouchDesigner connected. Total clients: {len(self.clients)}")
        
    async def disconnect(self, websocket):
        self.clients.remove(websocket)
        logger.info(f"TouchDesigner disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast_notes(self, notes):
        """
        Broadcast generated notes to all connected TouchDesigner clients.
        """
        if not self.clients:
            return
            
        message = {
            "type": "notes",
            "data": notes
        }
        
        serialized = json.dumps(message)
        to_remove = set()
        
        for client in self.clients:
            try:
                await client.send_text(serialized)
            except Exception as e:
                logger.error(f"Error broadcasting to TD: {e}")
                to_remove.add(client)
                
        for client in to_remove:
            self.clients.remove(client)

# Global instance
td_bridge = TouchDesignerBridge()

