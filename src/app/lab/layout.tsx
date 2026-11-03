import { AppSidebar } from "@/components/app-sidebar"
import { LabHeader } from "@/components/lab/lab-header"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"

export default function LabLayout({ children }: LayoutProps<"/lab">) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="min-w-0">
        <LabHeader />
        <div className="flex-1 px-4 pb-8 md:px-6">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  )
}
