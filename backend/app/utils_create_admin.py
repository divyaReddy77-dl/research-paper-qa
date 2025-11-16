from app.database import SessionLocal
from app import models, auth

def create_admin_user():
    db = SessionLocal()
    
    try:
        # Check if admin already exists
        admin = db.query(models.User).filter(models.User.username == "admin").first()
        if admin:
            print("Admin user already exists!")
            print("Username: admin")
            print("Password: admin123 (if not changed)")
            return
        
        # Create admin user
        hashed_password = auth.get_password_hash("admin123")
        admin_user = models.User(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        print("=" * 50)
        print("Admin user created successfully!")
        print("=" * 50)
        print("Username: admin")
        print("Password: admin123")
        print("=" * 50)
        print("\nPlease change the password after first login!")
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()